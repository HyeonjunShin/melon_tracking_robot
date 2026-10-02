import numpy as np
import multiprocessing as mp
from scipy.spatial.transform import Rotation as R

from lib.camera.gemini336 import Gemini336
from lib.camera.buffer import CameraShm

from shm_flange import FlangeShm, FlangeTF
from shm_target import TargetShm, TargetTF
from multiprocessing import shared_memory

from lib.detector.detection import Detector, compute_pose_with_kpts
from lib.detector.detection import DetectorBuffer
import cv2
import time
import os

np.set_printoptions(suppress=True)

# 카메라 파라미터
fx = 693.3102
fy = 693.4061
cx = 639.6599
cy = 365.0724

K = np.array(
    [
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ],
    dtype=np.float64,
)

D = np.array(
    [
        0.00743896747007966,
        -0.05456198751926422,
        0.03670734167098999,
        0.0,
        0.0,
        0.0,
        0.00016195396892726421,
        -0.001005938509479165,
    ],
    dtype=np.float32,
)
color_shape = (1280, 720, 3)
depth_shape = (1280, 720, 1)

TOOL_CAM = np.array(
    [
        [0.0, -0.93969, 0.34202, 0.06146],
        [1.0, 0.0, 0.0, 0.00100],
        [0.0, 0.34202, 0.93969, 0.03085],
        [0.0, 0.0, 0.0, 1.00000],
    ],
    dtype=np.float64,
)

TOOL_SCUTION = np.array(
    [
        [0.0, -1.0, 0.0, 0.000],
        [1.0, 0.0, 0.0, 0.000],
        [0.0, 0.0, 1.0, 0.255],
        [0.0, 0.0, 0.0, 1.000],
    ],
    dtype=np.float64,
)


def camera_runner(
    camera_shm_name,
    stop_signal,
    frame_ready_signal,
    color_shape=(1280, 720, 3),
    depth_shape=(1280, 720, 1),
):
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {0})

    camera_shm = CameraShm(
        shm_name=camera_shm_name,
        is_owner=False,
        color_shape=color_shape,
        depth_shape=depth_shape,
    )

    settings_path = "./gemini336_settings.json"
    camera = Gemini336(
        color_shape=color_shape, depth_shape=depth_shape, settings_path=settings_path
    )

    while not stop_signal.is_set():
        frame = camera.get_frame()
        if frame is None:
            camera_shm.status = False
            continue
        color = frame.get_color_frame()
        depth = frame.get_depth_frame()
        if color is None or depth is None:
            camera_shm.status = False
            continue
        camera_shm.status = True

        timestamp = depth.get_global_timestamp_us()
        color_data = color.get_data()
        depth_data = depth.get_data()

        camera_shm.write(timestamp, color_data, depth_data)
        frame_ready_signal.set()
        time.sleep(0.03)


def detector_runner(
    camera_shm_name,
    target_shm_name,
    flange_shm_name,
    stop_signal,
    frame_ready_signal,
):
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {1})

    # Detector 인스턴스 생성 (INT8 NMS-Free 모델)
    detector = Detector(model_path="./model_int8.xml", conf_threshold=0.25)

    camera_shm = CameraShm(shm_name=camera_shm_name, is_owner=False)
    flange_shm = FlangeShm(shm_name=flange_shm_name, is_owner=False)
    target_shm = TargetShm(shm_name=target_shm_name, is_owner=False)

    try:
        while not stop_signal.is_set():
            if not camera_shm.status:
                target_shm.status = False
                continue
            else:
                target_shm.status = True

            if not frame_ready_signal.wait(timeout=0.035):
                print("Camera timeout!!")
                continue
            frame_ready_signal.clear()

            current_frame = camera_shm.read()

            timestamp = current_frame.timestamp
            color = current_frame.color.copy()
            depth = current_frame.depth.copy()

            # 💡 1. OpenVINO INT8 포즈 추론 (Score, BBox, Keypoints)
            score, bbox, kpts = detector.detect(color)

            detected = False
            best_score = float(score)
            best_bbox = (
                bbox.copy() if bbox is not None else np.zeros((4,), dtype=np.float64)
            )
            best_TF = np.eye(4, dtype=np.float64)

            # 💡 2. 검출 성공 시 Body Centroid & Orientation 회전 행렬 계산
            if score > 0.25 and kpts is not None:
                centroid_cam, quat_cam = compute_pose_with_kpts(
                    depth_img=depth, kpts=kpts, K=K, DIST_COEFFS=D, vis_thresh=0.5
                )

                if centroid_cam is not None:
                    # 3. 카메라 좌표계 상의 $4 \times 4$ Transform Matrix 구성
                    rot_cam_mat = R.from_quat(
                        quat_cam
                    ).as_matrix()  # Quaternion -> 3x3 Rotation Matrix

                    T_cam = np.eye(4, dtype=np.float64)
                    T_cam[:3, :3] = rot_cam_mat
                    T_cam[:3, 3] = centroid_cam

                    # 4. Camera -> Tool (Flange) 변환
                    T_flange_target = TOOL_CAM @ T_cam

                    # 5. Tool (Flange) -> Base Robot Coordinate 변환
                    flange_pose = flange_shm.read(timestamp)
                    TF_FLANGE = flange_pose.TF.copy()

                    best_TF = TF_FLANGE @ T_flange_target
                    detected = True

            # 6. 로봇 컨트롤러 공유 메모리에 TF 전송
            target_shm.write(
                timestamp=timestamp,
                detected=detected,
                score=best_score,
                bbox=best_bbox.astype(np.float64),
                TF=best_TF,
            )

    finally:
        target_shm.status = False
        target_shm.close()


def main():
    mp.set_start_method("spawn", force=True)
    stop_signal = mp.Event()
    frame_ready_signal = mp.Event()

    flange_shm = FlangeShm(shm_name="flange_shm", is_owner=True)
    camera_shm = CameraShm(shm_name="camera_shm", is_owner=True)
    target_shm = TargetShm(shm_name="target_shm", is_owner=True)

    camera_process = mp.Process(
        target=camera_runner,
        args=(camera_shm.shm_name, stop_signal, frame_ready_signal),
    )
    camera_process.start()

    detector_process = mp.Process(
        target=detector_runner,
        args=(
            camera_shm.shm_name,
            target_shm.shm_name,
            flange_shm.shm_name,
            stop_signal,
            frame_ready_signal,
        ),
    )
    detector_process.start()

    try:
        while True:
            target = target_shm.read()
            if target.detected:
                timestamp_us = target.timestamp
                score = target.score
                target_TF = target.TF
                print(f"🎯 Melon Detected! Score: {score:.2f}")
                print(f"📍 Target Robot TF:\n{target_TF}\n")
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[알림] 사용자에 의해 Ctrl + C가 입력되었습니다.")

    finally:
        stop_signal.set()
        camera_process.join(timeout=3)
        if camera_process.is_alive():
            camera_process.terminate()
        detector_process.join(timeout=3)
        if detector_process.is_alive():
            detector_process.terminate()


if __name__ == "__main__":
    main()
