import numpy as np
import openvino as ov
import os
from lib.camera.buffer import CameraBuffer
from lib.detector.model import DetectionModel
from multiprocessing import shared_memory
from dataclasses import dataclass
import ctypes


class BufferHeader(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_bool),
        ("index", ctypes.c_int64),
        ("latency", ctypes.c_double),
    ]


@dataclass
class TrackObj:
    timestamp: int
    detected: bool
    score: float
    bbox: np.ndarray
    centroid: np.ndarray  # [X, Y, Z] (3D)
    rotation: np.ndarray  # [qx, qy, qz, qw] (4D Quaternion)


class DetectorBuffer:
    def __init__(
        self,
        shm_name: str,
        is_owner: bool = False,
    ):
        self.shm_name = shm_name
        self.is_owner = is_owner

        # Header: status(bool_ 1 byte -> 8 byte alignment) + write_index(int64 8 bytes)
        self.header_bytes = 8 + 8

        # Slot Layout (Bytes)
        self.timestamp_bytes = np.dtype(np.uint64).itemsize  # 8
        self.detected_bytes = np.dtype(np.bool_).itemsize  # 1
        # C-Struct Alignment 패딩 (1byte bool -> 8byte align을 위해 7bytes 패딩)
        self.detected_pad_bytes = 7
        self.score_bytes = np.dtype(np.float64).itemsize  # 8
        self.bbox_bytes = np.dtype(np.float64).itemsize * 4  # 32

        self.centroid_bytes = np.dtype(np.float64).itemsize * 3  # 24
        self.rotation_bytes = np.dtype(np.float64).itemsize * 4  # 32

        self.slot_bytes = (
            self.timestamp_bytes
            + self.detected_bytes
            + self.detected_pad_bytes
            + self.score_bytes
            + self.bbox_bytes
            + self.centroid_bytes
            + self.rotation_bytes
        )  # Total: 136 bytes

        self.total_bytes = self.header_bytes + (2 * self.slot_bytes)

        if self.is_owner:
            try:
                old_shm = shared_memory.SharedMemory(name=self.shm_name)
                old_shm.close()
                old_shm.unlink()
            except FileNotFoundError:
                pass
            self.shm = shared_memory.SharedMemory(
                name=self.shm_name,
                create=True,
                size=self.total_bytes,
            )
        else:
            self.shm = shared_memory.SharedMemory(name=self.shm_name, create=False)

        # Header Mapping
        self.status_arr = np.ndarray((1,), dtype=np.bool_, buffer=self.shm.buf, offset=0)
        self.write_index_arr = np.ndarray((1,), dtype=np.int64, buffer=self.shm.buf, offset=8)

        # Double Buffer Slots Mapping
        self.slots = []
        for i in range(2):
            slot_offset = self.header_bytes + (i * self.slot_bytes)

            timestamp_arr = np.ndarray((1,), dtype=np.uint64, buffer=self.shm.buf, offset=slot_offset)

            offset_curr = slot_offset + self.timestamp_bytes
            detected_arr = np.ndarray((1,), dtype=np.bool_, buffer=self.shm.buf, offset=offset_curr)

            offset_curr += self.detected_bytes + self.detected_pad_bytes
            score_arr = np.ndarray((1,), dtype=np.float64, buffer=self.shm.buf, offset=offset_curr)

            offset_curr += self.score_bytes
            bbox_arr = np.ndarray((4,), dtype=np.float64, buffer=self.shm.buf, offset=offset_curr)

            offset_curr += self.bbox_bytes
            centroid_arr = np.ndarray((3,), dtype=np.float64, buffer=self.shm.buf, offset=offset_curr)

            offset_curr += self.centroid_bytes
            rotation_arr = np.ndarray((4,), dtype=np.float64, buffer=self.shm.buf, offset=offset_curr)

            self.slots.append(
                {
                    "timestamp": timestamp_arr,
                    "detected": detected_arr,
                    "score": score_arr,
                    "bbox": bbox_arr,
                    "centroid": centroid_arr,
                    "rotation": rotation_arr,
                }
            )

    def write(
        self,
        timestamp: int,
        detected: bool,
        score: float = 0.0,
        bbox: np.ndarray = None,
        centroid: np.ndarray = None,
        rotation: np.ndarray = None,
    ):
        if bbox is None:
            bbox = np.zeros((4,), dtype=np.float64)
        if centroid is None:
            centroid = np.zeros((3,), dtype=np.float64)
        if rotation is None:
            rotation = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

        current_idx = int(self.write_index_arr[0])
        next_idx = 1 - current_idx
        target_slot = self.slots[next_idx]

        target_slot["timestamp"][0] = timestamp
        target_slot["detected"][0] = detected
        target_slot["score"][0] = score

        np.copyto(target_slot["bbox"], bbox.reshape(-1))
        np.copyto(target_slot["centroid"], centroid.reshape(-1))
        np.copyto(target_slot["rotation"], rotation.reshape(-1))

        self.write_index_arr[0] = next_idx

    def read_latest(self) -> TrackObj:
        current_idx = int(self.write_index_arr[0])
        slot = self.slots[current_idx]

        return TrackObj(
            timestamp=int(slot["timestamp"][0]),
            detected=bool(slot["detected"][0]),
            score=float(slot["score"][0]),
            bbox=slot["bbox"].copy(),
            centroid=slot["centroid"].copy(),
            rotation=slot["rotation"].copy(),
        )

    def set_status(self, is_good: bool):
        self.status_arr[0] = is_good

    def get_status(self) -> bool:
        return bool(self.status_arr[0])

    def close(self):
        del self.slots
        del self.status_arr
        del self.write_index_arr

        import gc

        gc.collect()

        self.shm.close()
        if self.is_owner:
            try:
                self.shm.unlink()
            except FileNotFoundError:
                pass


class Detector:
    def __init__(self):
        self.model = DetectionModel()
        self.conf_threshold = 0.8
        self.model_path = "model_int8.xml"

        core = ov.Core()
        device_name = "GPU" if "GPU" in core.available_devices else "CPU"
        print(f"📦 OpenVINO INT8 모델 로드 중... [디바이스: {device_name}]")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                1,
                "Not found the model file.",
                self.model_path,
            )

        ov_model = core.read_model(self.model_path)
        self.compiled_model = core.compile_model(ov_model, device_name)

        # Binding the output layer.
        self.output_0 = self.compiled_model.output(0)
        self.output_1 = self.compiled_model.output(1)

    def detect(self, color):  # (720, 1280, 3)
        input_tensor = color[None, ...]
        results = self.compiled_model({0: input_tensor})

        scores = results[self.output_0].squeeze()
        decode_bboxes = results[self.output_1][0]

        keep = scores > self.conf_threshold
        final_boxes = decode_bboxes[keep]
        final_scores = scores[keep]

        return (
            final_scores,
            final_boxes,
        )


import numpy as np


def compute_pose(depth, bbox, fx, fy, cx, cy, patch_size=30, min_depth=0.1, max_depth=3.0):
    """BBox의 중심점을 기준으로 patch_size x patch_size 패치를 추출하고, Depth 평균을 구하여 3D Centroid를

    계산합니다.

    좌표계 정의:
    - 오른쪽: +X
    - 화면 하단: +Y
    - 화면 깊이 방향 (아래쪽): +Z
    - Rotation: 고정 단위 쿼터니언 [0, 0, 0, 1]
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        return None, None

    # 1. BBox의 2D 중심점 계산 (픽셀 좌표)
    u_center = (x1 + x2) / 2.0
    v_center = (y1 + y2) / 2.0

    # 2. 중심점 기준 패치(Patch) 영역 좌표 산출 (예: 30x30 또는 50x50)
    half_patch = patch_size / 2.0
    px1 = int(max(0, np.floor(u_center - half_patch)))
    py1 = int(max(0, np.floor(v_center - half_patch)))
    px2 = int(min(depth.shape[1], np.ceil(u_center + half_patch)))
    py2 = int(min(depth.shape[0], np.ceil(v_center + half_patch)))

    if px2 <= px1 or py2 <= py1:
        return None, None

    # 3. Depth 패치 ROI 추출
    depth_patch = np.squeeze(depth[py1:py2, px1:px2])
    if depth_patch.size == 0:
        return None, None

    # 4. 유효 Depth 마스킹 (0 및 노이즈 제거)
    valid_mask = depth_patch > 0
    if not np.any(valid_mask):
        return None, None

    valid_depths = depth_patch[valid_mask].astype(np.float64)

    # 단위 자동 감지 (mm -> m 변환)
    if np.median(valid_depths) > 10.0:
        valid_depths /= 1000.0

    # 노이즈 범위(예: 0.1m ~ 3.0m) 필터링
    valid_depths = valid_depths[(valid_depths >= min_depth) & (valid_depths <= max_depth)]
    if len(valid_depths) == 0:
        return None, None

    # 5. 패치 영역의 Depth 평균(Mean) 계산
    zc = np.mean(valid_depths)

    # 6. Pinhole Model로 3D Centroid 계산 (m 단위)
    # X: 오른쪽 (+), Y: 하단 (+), Z: 화면 앞쪽/깊이 (+)
    xc = (u_center - cx) * zc / fx
    yc = (v_center - cy) * zc / fy

    centroid = np.array([xc, yc, zc], dtype=np.float64)

    # 7. 고정 쿼터니언 지정 (Roll=0, Pitch=0, Yaw=0 -> [qx=0, qy=0, qz=0, qw=1])
    rotation = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    return centroid, rotation

    # print(False if len(final_scores) == 0 else True)
    # print(final_scores)
    # print(final_bboxes)

    # measured_3d = None
    # if len(final_bboxes) > 0 and len(final_scores) > 0:
    #     final_bboxes[:, [0, 2]] = final_bboxes[:, [0, 2]] * 2
    #     final_bboxes[:, [1, 3]] = (final_bboxes[:, [1, 3]] - 12) * 2

    #     best_idx = np.argmax(final_scores)
    #     best_box = final_bboxes[best_idx].tolist()

    #     # Depth 기반 3D Centroid 측정
    #     measured_3d = kf_tracker.compute_3d_centroid(
    #         depth, best_box, FX, FY, CX, CY
    #     )

    #     x1, y1, x2, y2 = map(int, best_box)
    #     print(x1, y1, x2, y2)

    # state, is_updated = kf_tracker.update(measured_3d)

    # if kf_tracker.is_initialized:
    #     xc, yc, zc, vx, vy, vz = state
    #     detector_buffer.write(prev_ts, np.array([xc, yc, zc, vx, vy, vz], dtype=np.float64))
    #     detector_buffer.set_status(True)
    # pos_text = f"TF [X:{xc:.2f}, Y:{yc:.2f}, Z:{zc:.2f}] mm"
    # vel_text = f"V [Vx:{vx:.2f}, Vy:{vy:.2f}, Vz:{vz:.2f}] mm/s"
    # print(pos_text)
    # print(vel_text)


import cv2


def compute_pose_with_undistort(depth, bbox, K, DIST_COEFFS, patch_size=30):
    xmin, ymin, xmax, ymax = bbox

    u = (xmin + xmax) / 2.0
    v = (ymin + ymax) / 2.0

    h, w = depth.shape[:2]
    if not (0 <= u < w and 0 <= v < h):
        return None, None

    u_idx, v_idx = int(round(u)), int(round(v))
    half_patch = patch_size // 2

    v_min, v_max = max(0, v_idx - half_patch), min(h, v_idx + half_patch)
    u_min, u_max = max(0, u_idx - half_patch), min(w, u_idx + half_patch)

    depth_patch = depth[v_min:v_max, u_min:u_max]
    valid_depths = depth_patch[depth_patch > 0]

    if len(valid_depths) == 0:
        return None, None

    z_m = float(np.median(valid_depths))
    z_m /= 1000.0  # Convert unit mm to m

    pixel_pt = np.array([[[u, v]]], dtype=np.float32)

    undistorted_pt = cv2.undistortPoints(pixel_pt, K, DIST_COEFFS, P=K)
    u_undist, v_undist = undistorted_pt[0][0]

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    x_cam = (u_undist - cx) * z_m / fx
    y_cam = (v_undist - cy) * z_m / fy
    z_cam = z_m

    centroid_cam = np.array([x_cam, y_cam, z_cam], dtype=np.float64)
    rotation_cam = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return centroid_cam, rotation_cam
