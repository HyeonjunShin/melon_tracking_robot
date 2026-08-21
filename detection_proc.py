import numpy as np
import multiprocessing as mp
from scipy.spatial.transform import Rotation as R

from lib.camera.gemini336 import Gemini336
from lib.camera.buffer import CameraBuffer

from lib.detector.detection import Detector, compute_pose, compute_pose_with_undistort
from lib.detector.detection import DetectorBuffer

T_FLANGE_CAMERA = np.array(
    [
        [0.0, -0.93969262, 0.34202014, 0.05991575],
        [1.0, 0.0, 0.0, 0.00358377],
        [-0.0, 0.34202014, 0.93969262, 0.03416166],
        [0.0, 0.0, 0.0, 1.0],
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
    buffer = CameraBuffer(
        shm_name=camera_shm_name,
        is_owner=False,
        color_shape=color_shape,
        depth_shape=depth_shape,
    )

    # current_dir = Path(__file__).resolve().parent
    # settings_path = str((current_dir / "gemini336_settings.json").resolve())
    settings_path = "./gemini336_settings.json"
    camera = Gemini336(color_shape=color_shape, depth_shape=depth_shape, settings_path=settings_path)

    while not stop_signal.is_set():
        # loop_start = time.perf_counter()

        frame = camera.get_frame()
        if frame is None:
            buffer.set_status(False)
            continue
        color = frame.get_color_frame()
        depth = frame.get_depth_frame()
        if color is None or depth is None:
            buffer.set_status(False)
            continue
        buffer.set_status(True)

        timestamp = depth.get_global_timestamp_us()
        color_data = color.get_data()
        depth_data = depth.get_data()
        # process_end = time.perf_counter()

        buffer.write(timestamp, color_data, depth_data)
        frame_ready_signal.set()

        # loop_end = time.perf_counter()
        # proc_time_ms = (process_end - loop_start) * 1000
        # total_time_ms = (loop_end - loop_start) * 1000

        # print(
        #     f"[카메라 처리]: {proc_time_ms:.2f} ms | [전체 루프]: {total_time_ms:.2f} ms (FPS: {1000/total_time_ms:.1f})"
        # )

def transform_cam_to_flange(centroid_cam, rotation_cam):
    T_cam_obj = np.eye(4, dtype=np.float64)

    T_cam_obj[:3, :3] = R.from_quat(rotation_cam).as_matrix()
    T_cam_obj[:3, 3] = centroid_cam

    T_flange_obj = T_FLANGE_CAMERA @ T_cam_obj

    centroid_flange = T_flange_obj[:3, 3]
    rotation_flange = R.from_matrix(T_flange_obj[:3, :3]).as_quat()

    return centroid_flange, rotation_flange

def detector_runner(
    camera_shm_name,
    detector_shm_name,
    stop_signal,
    frame_ready_signal,
):
    camera_buffer = CameraBuffer(shm_name=camera_shm_name, is_owner=False)
    detector_buffer = DetectorBuffer(shm_name=detector_shm_name, is_owner=False)

    detector = Detector()

    FX, FY = 693.3102, 693.4061
    CX, CY = 639.6599, 365.0724

    K = np.array(
        [
            [FX, 0.0, CX],
            [0.0, FY, CY],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    DIST_COEFFS = np.array(
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
        dtype=np.float64,
    )

    try:
        while not stop_signal.is_set():
            if not frame_ready_signal.wait(timeout=0.035):
                print("Camera timeout!!")
                continue
            frame_ready_signal.clear()

            # loop_start = time.perf_counter()

            if not camera_buffer.get_status():
                detector_buffer.set_status(False)
                continue
            else:
                detector_buffer.set_status(True)

            current_frame = camera_buffer.read_latest_frame()

            # if prev_ts == current_frame.timestamp:
            # continue
            # prev_ts = current_frame.timestamp
            # frame_time_sec = prev_ts / 1_000_000.0

            timestamp = current_frame.timestamp
            color = current_frame.color.copy()
            depth = current_frame.depth.copy()

            scores, bboxes = detector.detect(color)  # detecte the objects.

            detected = False
            best_score = 0.0
            best_bbox = np.zeros((4,), dtype=np.float64)
            best_centroid = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            best_rotation = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

            if len(scores) > 0:
                best_idx = np.argmax(scores)
                best_score = float(scores[best_idx])
                best_bbox = bboxes[best_idx].copy()

                best_bbox[[0, 2]] = best_bbox[[0, 2]] * 2
                best_bbox[[1, 3]] = (best_bbox[[1, 3]] - 12) * 2

                centroid_camera, rotation_camera = compute_pose_with_undistort(
                    depth, best_bbox, K, DIST_COEFFS, patch_size=30
                )
                if centroid_camera is not None:
                    best_centroid, best_rotation = transform_cam_to_flange(centroid_camera, rotation_camera)
                    detected = True

            detector_buffer.write(
                timestamp=timestamp,
                detected=detected,
                score=best_score,
                bbox=best_bbox.astype(np.float64),
                centroid=best_centroid,
                rotation=best_rotation,
            )
            # loop_end = time.perf_counter()
            # proc_time_ms = (loop_end - loop_start) * 1000
            # print(proc_time_ms)
    finally:
        detector_buffer.set_status(False)
        detector_buffer.close()

def main():
    mp.set_start_method("spawn", force=True)
    stop_signal = mp.Event()
    frame_ready_signal = mp.Event()

    camera_shm_name = "camera_buffer"
    camera_buffer = CameraBuffer(shm_name=camera_shm_name, is_owner=True)
    camera_process = mp.Process(target=camera_runner, args=(camera_shm_name, stop_signal, frame_ready_signal))
    camera_process.start()



    while not stop_signal.is_set():
        if not frame_ready_signal.wait(0.05):
            continue
        frame_ready_signal.clear()

        if not camera_buffer.get_status():
            



if __name__ == "__main__":
    main()
