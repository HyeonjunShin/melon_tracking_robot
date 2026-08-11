import numpy as np
import openvino as ov
import os
from lib.camera.camera_shm import CameraBuffer
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
    pose: np.ndarray


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
        self.pose_bytes = np.dtype(np.float64).itemsize * 7  # 56

        self.slot_bytes = (
            self.timestamp_bytes
            + self.detected_bytes
            + self.detected_pad_bytes
            + self.score_bytes
            + self.bbox_bytes
            + self.pose_bytes
        )  # Total: 112 bytes

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
            self.shm = shared_memory.SharedMemory(
                name=self.shm_name, create=False
            )

        # Header Mapping
        self.status_arr = np.ndarray(
            (1,), dtype=np.bool_, buffer=self.shm.buf, offset=0
        )
        self.write_index_arr = np.ndarray(
            (1,), dtype=np.int64, buffer=self.shm.buf, offset=8
        )

        # Double Buffer Slots Mapping
        self.slots = []
        for i in range(2):
            slot_offset = self.header_bytes + (i * self.slot_bytes)

            timestamp_arr = np.ndarray(
                (1,), dtype=np.uint64, buffer=self.shm.buf, offset=slot_offset
            )

            offset_curr = slot_offset + self.timestamp_bytes
            detected_arr = np.ndarray(
                (1,), dtype=np.bool_, buffer=self.shm.buf, offset=offset_curr
            )

            offset_curr += self.detected_bytes + self.detected_pad_bytes
            score_arr = np.ndarray(
                (1,), dtype=np.float64, buffer=self.shm.buf, offset=offset_curr
            )

            offset_curr += self.score_bytes
            bbox_arr = np.ndarray(
                (4,),
                dtype=np.float64,
                buffer=self.shm.buf,
                offset=offset_curr,  # (5,) -> (4,) 수정 완료
            )

            offset_curr += self.bbox_bytes
            pose_arr = np.ndarray(
                (7,), dtype=np.float64, buffer=self.shm.buf, offset=offset_curr
            )

            self.slots.append(
                {
                    "timestamp": timestamp_arr,
                    "detected": detected_arr,
                    "score": score_arr,
                    "bbox": bbox_arr,
                    "pose": pose_arr,
                }
            )

    def write(
        self,
        timestamp: int,
        detected: bool,
        score: float = 0.0,
        bbox: np.ndarray = None,
        pose: np.ndarray = None,
    ):
        if bbox is None:
            bbox = np.zeros((4,), dtype=np.float64)
        if pose is None:
            pose = np.zeros((7,), dtype=np.float64)

        current_idx = int(self.write_index_arr[0])
        next_idx = 1 - current_idx
        target_slot = self.slots[next_idx]

        target_slot["timestamp"][0] = timestamp
        target_slot["detected"][0] = detected
        target_slot["score"][0] = score

        # 안전한 1D 배열 복사
        np.copyto(target_slot["bbox"], bbox.reshape(-1))
        np.copyto(target_slot["pose"], pose.reshape(-1))

        # Write Index 스위칭 (Atomic-like operation)
        self.write_index_arr[0] = next_idx

    def read_latest(self) -> TrackObj:
        current_idx = int(self.write_index_arr[0])
        slot = self.slots[current_idx]

        return TrackObj(
            timestamp=int(slot["timestamp"][0]),
            detected=bool(slot["detected"][0]),
            score=float(slot["score"][0]),
            bbox=slot["bbox"].copy(),
            pose=slot["pose"].copy(),
        )

    def set_status(self, is_good: bool):
        self.status_arr[0] = is_good

    def get_status(self) -> bool:
        return bool(self.status_arr[0])

    def close(self):
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


def compute_6d_pose(depth, bbox, fx, fy, cx, cy, crop_ratio=0.6):
    """
    BBox 중앙 60% ROI 영역의 Depth 데이터를 3D Centroid(X, Y, Z)로 복원 후
    카메라 정면 고정 회전 쿼터니언 [0, 0, 0, 1]을 부여하여 반환하는 함수 (프로토타입용)
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        return None

    # 1. 중앙 60% 영역 Crop
    scale = np.sqrt(crop_ratio)
    margin_w = (w * (1 - scale)) / 2
    margin_h = (h * (1 - scale)) / 2

    cx1 = int(max(0, x1 + margin_w))
    cy1 = int(max(0, y1 + margin_h))
    cx2 = int(min(depth.shape[1], x2 - margin_w))
    cy2 = int(min(depth.shape[0], y2 - margin_h))

    if cx2 <= cx1 or cy2 <= cy1:
        return None

    # Depth ROI 추출 및 2D 차원 보장
    depth_roi = np.squeeze(depth[cy1:cy2, cx1:cx2])
    if depth_roi.ndim != 2:
        return None

    # 2. 유효 Depth 마스킹 (0 및 비정상 센서 값 제외)
    # 단위가 mm일 경우 (100mm ~ 3000mm) / m일 경우 (0.1m ~ 3.0m)
    # 아래는 m 단위 기준 (센서 데이터가 mm라면 > 100 조건으로 수정)
    valid_mask = depth_roi > 0
    if not np.any(valid_mask):
        return None

    # 3. Depth 노이즈 튀는 현상 방지 (Median/Percentile 필터링)
    valid_depths = depth_roi[valid_mask].astype(np.float64)

    # 단위 변환 (mm -> m 변환이 필요한 경우)
    if np.median(valid_depths) > 10.0:  # mm 단위 데이터 감지 시
        valid_depths /= 1000.0

    # 4. Pixel Coordinates Grid 생성
    u_grid, v_grid = np.meshgrid(np.arange(cx1, cx2), np.arange(cy1, cy2))
    u_valid = u_grid[valid_mask]
    v_valid = v_grid[valid_mask]

    # 5. Pinhole Model로 3D 좌표 복원 (m 단위)
    z_pts = valid_depths
    x_pts = (u_valid - cx) * z_pts / fx
    y_pts = (v_valid - cy) * z_pts / fy

    # 6. 아웃라이어 제거 후 3D Centroid 계산 (np.median 활용)
    xc = np.median(x_pts)
    yc = np.median(y_pts)
    zc = np.median(z_pts)
    centroid = np.array([xc, yc, zc], dtype=np.float64)

    # 7. 고정 쿼터니언 지정 (Roll=0, Pitch=0, Yaw=0 -> [qx=0, qy=0, qz=0, qw=1])
    fixed_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    # [X, Y, Z, 0, 0, 0, 1] 7D Vector 반환
    return np.concatenate([centroid, fixed_quat])


def detector_runner(
    camera_shm_name,
    detector_shm_name,
    stop_signal,
):
    import time
    import numpy as np
    from lib.tracker.tracker import (
        KalmanFilter3D,
    )

    camera_buffer = CameraBuffer(shm_name=camera_shm_name, is_owner=False)
    detector_buffer = DetectorBuffer(shm_name=detector_shm_name, is_owner=False)

    detector = Detector()
    FX, FY = (693.3102, 693.4061)
    CX, CY = (639.6599, 365.0724)

    prev_ts = 0
    try:
        while not stop_signal.is_set():
            if not camera_buffer.get_status():
                time.sleep(0.001)
                detector_buffer.set_status(False)
                continue
            else:
                detector_buffer.set_status(True)

            current_frame = camera_buffer.read_latest_frame()
            if prev_ts == current_frame.timestamp:
                continue

            prev_ts = current_frame.timestamp
            color = current_frame.color
            depth = current_frame.depth

            scores, bboxes = detector.detect(color)

            if len(scores) == 0:
                detector_buffer.write(
                    timestamp=prev_ts,
                    detected=False,
                    score=0.0,
                    bbox=np.zeros((4,), dtype=np.float64),
                    pose=np.zeros((7,), dtype=np.float64),
                )
                continue

            best_idx = np.argmax(scores)
            best_score = float(scores[best_idx])
            best_bbox = bboxes[best_idx].copy()

            best_bbox[[0, 2]] = best_bbox[[0, 2]] * 2
            best_bbox[[1, 3]] = (best_bbox[[1, 3]] - 12) * 2

            pose_7d = compute_6d_pose(
                depth, best_bbox, FX, FY, CX, CY, crop_ratio=0.6
            )
            if pose_7d is None:
                detector_buffer.write(
                    timestamp=prev_ts,
                    detected=False,
                    score=best_score,
                    bbox=best_bbox.astype(np.float64),
                    pose=np.zeros((7,), dtype=np.float64),
                )
                continue

            detector_buffer.write(
                timestamp=prev_ts,
                detected=True,
                score=best_score,
                bbox=best_bbox.astype(np.float64),
                pose=pose_7d.astype(np.float64),
            )

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
    finally:
        detector_buffer.set_status(False)
        detector_buffer.close()
