import os
import sys
import time
import cv2
import numpy as np
import multiprocessing as mp
from scipy.spatial.transform import Rotation as R

from lib.camera.gemini336 import Gemini336
from lib.camera.buffer import CameraShm
from shm_flange import FlangeShm, FlangeTF
from shm_target import TargetShm, TargetTF
from multiprocessing import shared_memory

from lib.detector.detection import Detector, compute_pose, compute_pose_with_undistort
from lib.detector.detection import DetectorBuffer

np.set_printoptions(suppress=True)

# 카메라 파라미터
fx, fy = 693.3102, 693.4061
cx, cy = 639.6599, 365.0724
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
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


def setup_save_directory(base_path="/media/uon/Samsung USB/melon_pack"):
    """시작 시 USB 저장 위치를 확인하고 rgb/depth 폴더를 생성합니다."""
    print(f"\n[저장 위치 확인 중]: {base_path}")

    if not os.path.exists(base_path):
        print(f"❌ [오류] 저장 경로가 존재하지 않습니다: {base_path}")
        print("USB 마운트 상태를 확인하거나 경로를 올바르게 입력했는지 검토하세요.")
        sys.exit(1)

    rgb_dir = os.path.join(base_path, "rgb")
    depth_dir = os.path.join(base_path, "depth")

    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    print(f"✅ [준비 완료] 이미지 저장 폴더:")
    print(f"  - RGB   : {rgb_dir}")
    print(f"  - Depth : {depth_dir}\n")
    return rgb_dir, depth_dir


def camera_runner(
    camera_shm_name, stop_signal, frame_ready_signal, color_shape=(1280, 720, 3), depth_shape=(1280, 720, 1)
):
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {0})

    camera_shm = CameraShm(
        shm_name=camera_shm_name, is_owner=False, color_shape=color_shape, depth_shape=depth_shape
    )
    settings_path = "./gemini336_config_lab.json"
    camera = Gemini336(color_shape=color_shape, depth_shape=depth_shape, settings_path=settings_path)

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


def detector_runner(camera_shm_name, target_shm_name, flange_shm_name, stop_signal, frame_ready_signal):
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {1})
    detector = Detector()

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
                continue
            frame_ready_signal.clear()

            current_frame = camera_shm.read()
            timestamp = current_frame.timestamp
            color = current_frame.color.copy()
            depth = current_frame.depth.copy()

            scores, bboxes = detector.detect(color)

            detected = False
            best_score = 0.0
            best_bbox = np.zeros((4,), dtype=np.float64)
            best_TF = np.eye(4)

            if len(scores) > 0:
                best_idx = np.argmax(scores)
                best_score = float(scores[best_idx])
                best_bbox = bboxes[best_idx].copy()

                best_bbox[[0, 2]] = best_bbox[[0, 2]] * 2
                best_bbox[[1, 3]] = (best_bbox[[1, 3]] - 12) * 2

                xmin, ymin, xmax, ymax = best_bbox
                u = (xmin + xmax) / 2.0
                v = (ymin + ymax) / 2.0
                u_idx, v_idx = int(round(u)), int(round(v))

                half_patch = 30 // 2
                v_min, v_max = max(0, v_idx - half_patch), min(color_shape[1], v_idx + half_patch)
                u_min, u_max = max(0, u_idx - half_patch), min(color_shape[0], u_idx + half_patch)

                depth_patch = depth[v_min:v_max, u_min:u_max]
                valid_depths = depth_patch[depth_patch > 0]

                if len(valid_depths) > 0:
                    Z = float(np.median(valid_depths)) * 0.001
                    pixel_pt = np.array([[[u, v]]], dtype=np.float32)
                    undistorted_pt = cv2.undistortPoints(pixel_pt, K, D).squeeze()
                    X, Y = undistorted_pt * Z
                    point_3d_camera = np.array([X, Y, Z, 1.0], dtype=np.float64)

                    point_3d_flange = TOOL_CAM @ point_3d_camera
                    flange_pose = flange_shm.read(timestamp)
                    TF_FLANGE = flange_pose.TF.copy()

                    point_3d_robot = TF_FLANGE @ point_3d_flange
                    best_TF[:3, 3] = point_3d_robot[:3]
                    detected = True

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
    # 1. 저장 경로 확인 및 생성
    rgb_dir, depth_dir = setup_save_directory("/media/uon/Samsung USB/melon_pack")

    mp.set_start_method("spawn", force=True)
    stop_signal = mp.Event()
    frame_ready_signal = mp.Event()

    flange_shm = FlangeShm(shm_name="flange_shm", is_owner=True)
    camera_shm = CameraShm(shm_name="camera_shm", is_owner=True)
    target_shm = TargetShm(shm_name="target_shm", is_owner=True)

    camera_process = mp.Process(
        target=camera_runner, args=(camera_shm.shm_name, stop_signal, frame_ready_signal)
    )
    camera_process.start()

    detector_process = mp.Process(
        target=detector_runner,
        args=(camera_shm.shm_name, target_shm.shm_name, flange_shm.shm_name, stop_signal, frame_ready_signal),
    )
    detector_process.start()

    # 데이터 수집 관련 주요 변수 및 임계값
    is_recording = False
    save_count = 0
    prev_gray = None

    # [설정 1] 선명도 태깅 임계값 (라플라시안 분산)
    BLUR_THRESHOLD = 100.0

    # [설정 2] 카메라 이동 임계값 (중복 프레임 저장 방지)
    MOVEMENT_THRESHOLD = 8.0

    try:
        print("==================================================")
        print("  [SPACEBAR] : 자동 데이터 수집 시작 / 일시정지")
        print("  [q]        : 종료")
        print("  [수집 정책] : 모션 블러 포함, 화면 변화 감지 시 자동 저장")
        print(f"  [변화량 기준] : Frame Diff >= {MOVEMENT_THRESHOLD}")
        print("==================================================\n")

        while True:
            frame = camera_shm.read()
            timestamp_us = frame.timestamp
            color = frame.color.copy()
            depth = frame.depth.copy()

            # 1. 계산용 그레이스케일 및 시각화용 이미지 준비
            current_gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
            view = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

            # 2. 선명도 측정 (라플라시안 분산)
            sharpness = cv2.Laplacian(current_gray, cv2.CV_64F).var()
            is_sharp = sharpness >= BLUR_THRESHOLD
            blur_tag = "sharp" if is_sharp else "blur"

            # 3. 이전 프레임 대비 변화량 계산 (Mean Absolute Difference)
            motion_score = 0.0
            if prev_gray is not None:
                frame_diff = cv2.absdiff(current_gray, prev_gray)
                motion_score = float(np.mean(frame_diff))

            # Detector 시각화 (화면 출력용)
            target = target_shm.read()
            if target.detected:
                bbox = target.bbox
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    view,
                    f"Score: {target.score:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            # 4. 수집 조건 충족 시 저장 (녹화 활성화 & 카메라 이동 발생)
            if is_recording and (motion_score >= MOVEMENT_THRESHOLD):
                file_stem = f"melon_{timestamp_us}_{blur_tag}"

                rgb_path = os.path.join(rgb_dir, f"{file_stem}.png")
                depth_png_path = os.path.join(depth_dir, f"{file_stem}.png")

                # BBox/글씨 없는 순수 원본 RGB 이미지 저장 (RGB -> BGR 변환)
                raw_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                cv2.imwrite(rgb_path, raw_bgr)
                cv2.imwrite(depth_png_path, depth)

                save_count += 1
                print(
                    f"📸 [{save_count}] 저장 ({blur_tag.upper()}) | Sharpness: {sharpness:.1f} | Diff: {motion_score:.1f} -> {file_stem}"
                )

                # 기준 프레임 갱신
                prev_gray = current_gray.copy()

            elif prev_gray is None:
                prev_gray = current_gray.copy()

            # 5. UI 가이드 표시
            status_text = (
                f"REC | Saved: {save_count} | Sharp: {sharpness:.0f} ({blur_tag.upper()}) | Diff: {motion_score:.1f}"
                if is_recording
                else f"PAUSED | Saved: {save_count}"
            )
            status_color = (0, 255, 0) if is_recording else (255, 255, 0)

            cv2.putText(view, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            cv2.imshow("color", view)
            key = cv2.waitKey(1)

            # [SPACEBAR] : 수집 시작 / 일시정지 토글
            if key == 32:
                is_recording = not is_recording
                state_str = "시작" if is_recording else "일시정지"
                print(f"\n▶️ [자동 수집 {state_str}]")

            elif key == ord("q"):
                break

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
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
