import numpy as np
import multiprocessing as mp
from scipy.spatial.transform import Rotation as R

from lib.camera.gemini336 import Gemini336
from lib.camera.buffer import CameraShm

# from lib.control.flange_buffer import FlangeBuffer
from shm_flange import FlangeShm, FlangeTF
from shm_target import TargetShm, TargetTF
from multiprocessing import shared_memory

from lib.detector.detection import Detector, compute_pose, compute_pose_with_undistort
from lib.detector.detection import DetectorBuffer
import cv2
import time
import os

np.set_printoptions(suppress=True)

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
    camera = Gemini336(color_shape=color_shape, depth_shape=depth_shape, settings_path=settings_path)

    while not stop_signal.is_set():
        # loop_start = time.perf_counter()

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
        # process_end = time.perf_counter()

        camera_shm.write(timestamp, color_data, depth_data)
        frame_ready_signal.set()
        time.sleep(0.03)

    # loop_end = time.perf_counter()
    # proc_time_ms = (process_end - loop_start) * 1000
    # total_time_ms = (loop_end - loop_start) * 1000

    # print(
    #     f"[카메라 처리]: {proc_time_ms:.2f} ms | [전체 루프]: {total_time_ms:.2f} ms (FPS: {1000/total_time_ms:.1f})"
    # )


def detector_runner(
    camera_shm_name,
    target_shm_name,
    flange_shm_name,
    stop_signal,
    frame_ready_signal,
):
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
                # detector_buffer.set_status(True)
                target_shm.status = True

            if not frame_ready_signal.wait(timeout=0.035):
                print("Camera timeout!!")
                continue
            frame_ready_signal.clear()

            # loop_start = time.perf_counter()
            current_frame = camera_shm.read()

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
            best_TF = np.eye(4)
            # best_centroid = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            # best_rotation = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

            if len(scores) > 0:
                best_idx = np.argmax(scores)
                best_score = float(scores[best_idx])
                best_bbox = bboxes[best_idx].copy()

                best_bbox[[0, 2]] = best_bbox[[0, 2]] * 2
                best_bbox[[1, 3]] = (best_bbox[[1, 3]] - 12) * 2

                # centroid_camera, rotation_camera = compute_pose_with_undistort(
                #     depth, best_bbox, K, D, patch_size=30
                # )
                # if centroid_camera is not None:
                #     best_centroid, best_rotation = transform_cam_to_flange(centroid_camera, rotation_camera)
                #     detected = True

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
                    # pixel_pt = np.array([u, v], dtype=np.float32)

                    undistorted_pt = cv2.undistortPoints(pixel_pt, K, D).squeeze()
                    X, Y = undistorted_pt * Z
                    point_3d_camera = np.array([X, Y, Z, 1.0], dtype=np.float64)

                    # 1) Camera -> Tool (Flange) Transformation
                    point_3d_flange = TOOL_CAM @ point_3d_camera

                    # 2) Tool (Flange) -> Base Robot Coordinate Transformation
                    flange_pose = flange_shm.read(timestamp)
                    TF_FLANGE = flange_pose.TF.copy()

                    point_3d_robot = TF_FLANGE @ point_3d_flange
                    # point_3d_robot[2] += 0.5

                    best_TF[:3, 3] = point_3d_robot[:3]
                    detected = True

            target_shm.write(
                timestamp=timestamp,
                detected=detected,
                score=best_score,
                bbox=best_bbox.astype(np.float64),
                TF=best_TF,
            )
            # loop_end = time.perf_counter()
            # proc_time_ms = (loop_end - loop_start) * 1000
            # print(proc_time_ms)
    finally:
        target_shm.status = False
        target_shm.close()


mouse_x, mouse_y = 0, 0


def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    mouse_x, mouse_y = x, y


def main():
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

    # SHM_NAME = "control_buf"
    # try:
    #     old_shm = shared_memory.SharedMemory(name=SHM_NAME)
    #     old_shm.close()
    #     old_shm.unlink()
    # except FileNotFoundError:
    #     pass
    # SIZE_IN_BYTES = 7 * np.dtype(np.float64).itemsize
    # control_shm = shared_memory.SharedMemory(create=True, size=SIZE_IN_BYTES, name=SHM_NAME)
    # control_buf = np.ndarray((7,), np.float64, buffer=control_shm.buf)

    # flange_buffer = FlangeBuffer(shm_name="flange_pose", is_owner=True)

    # cv2.namedWindow("color")
    # cv2.setMouseCallback("color", mouse_callback)
    # state = 0

    try:
        while True:
            # frame = camera_shm.read()
            # timestamp_us = frame.timestamp
            # color = frame.color.copy()
            # depth = frame.depth.copy()

            # Z = depth[mouse_y][mouse_x].squeeze() * 0.001

            # view = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            # if Z > 0:
            #     u_distorted = float(mouse_x)
            #     v_distorted = float(mouse_y)
            #     point_pixel = np.array([u_distorted, v_distorted], dtype=np.float64)

            #     point_norm_camera = cv2.undistortPoints(point_pixel, K, D).squeeze()
            #     X, Y = point_norm_camera * Z

            #     point_3d_camera = np.array([X, Y, Z, 1], dtype=np.float32)
            #     point_3d_flange = TOOL_CAM @ point_3d_camera

            #     # TF_FLANGE = flange_buffer.get_flange_tf(frame.timestamp)["T_base_flange"]
            #     TF_FLANGE = flange_shm.read(frame.timestamp).TF
            #     #     # time_diff_us = flange_buffer.get_flange_tf(frame.timestamp)["time_diff_us"]
            #     #     # print(time_diff_us)
            #     #     TF_FLANGE[:3, 3] *= 0.001
            #     point_3d_robot = TF_FLANGE @ point_3d_flange
            #     # print(point_3d_robot)
            #     # target_tf = np.eye(4)
            #     #     target_tf[1, 1] *= -1
            #     #     target_tf[2, 2] *= -1
            #     #     target_tf[:, 3] = point_3d_robot
            #     #     # print(target_tf)

            #     #     # print(frame.timestamp, mouse_x, mouse_y)
            #     #     # print(X, Y, Z)
            #     #     # print(point_3d_flange[0], point_3d_flange[1], point_3d_flange[2])
            #     #     print(point_3d_robot[0], point_3d_robot[1], point_3d_robot[2])

            #     #     # text = f"3D: X={X:.1f}mm, Y={Y:.1f}mm, Z={Z:.1f}mm"
            #     #     # cv2.putText(view, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # cv2.circle(view, (mouse_x, mouse_y), 5, (0, 0, 255), -1)
            # flange_TF = flange_shm.read(time.time() * 1e-4).TF
            # print(flange_TF @ TOOL_SCUTION)
            # print(flange_TF)

            target = target_shm.read()
            if target.detected:
                timestamp_us = target.timestamp
                bbox = target.bbox
                score = target.score
                target_TF = target.TF
                print(target_TF)
            time.sleep(1)

            #     x1, y1, x2, y2 = map(int, bbox)
            #     label_text = f"Melon: {score:.2f}"
            #     ts_text = f"TS: {timestamp_us} us"

            #     cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 0), 2)

            #     (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            #     cv2.rectangle(
            #         view, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 255, 0), -1
            #     )  # 채워진 사각형
            #     cv2.putText(view, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            #     cv2.putText(view, ts_text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # if state == 2:
            #     obj = detector_buffer.get_latest_detection()
            #     control_buf[0] = obj.centroid[0]
            #     control_buf[1] = obj.centroid[1]
            #     control_buf[2] = obj.centroid[2]

            # cv2.imshow("color", view)
            # key = cv2.waitKey(1)
            # if key == ord("q"):
            # break
            # if key == ord("c"):
            #     control_buf[6] = 0
            #     state = 0
            # if key == 32:
            #     control_buf[6] = 1
            #     control_buf[0] = point_3d_robot[0]
            #     control_buf[1] = point_3d_robot[1]
            #     control_buf[2] = point_3d_robot[2]
            #     # solver.set_end_effector_offset(TOOL_SCUTION)
            #     # TARGET_POSE = [
            #     #     point_3d_robot[0],
            #     #     point_3d_robot[1],
            #     #     point_3d_robot[2],
            #     #     180.0,
            #     #     180.0,
            #     #     0,
            #     # ]
            #     # print(point_3d_robot[0], point_3d_robot[1], point_3d_robot[2])
            #     # state = 1

            # if key == ord("a"):
            #     control_buf[6] = 2
            #     state = 2

            # if state == 0:
            # solver.set_end_effector_offset(TOOL_CAM)
            # TARGET_POSE = INIT_POSE
            # if state == 1:
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

    # color_img = color.get_data().reshape(color_shape)

    # new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, D, (width), alpha=0)

    # undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # alpha=0 적용 시 튀어나온 검은 여백을 깔끔하게 크롭
    # x, y, w_box, h_box = roi
    # undistorted_img = undistorted_img[y : y + h_box, x : x + w_box]
    # import time
    # import cv2

    # mp.set_start_method("spawn", force=True)
    # stop_signal = mp.Event()
    # frame_ready_signal = mp.Event()

    # camera_shm_name = "camera_buffer"
    # camera_buffer = CameraBuffer(shm_name=camera_shm_name, is_owner=True)
    # camera_process = mp.Process(target=camera_runner, args=(camera_shm_name, stop_signal, frame_ready_signal))
    # camera_process.start()

    # detector_shm_name = "detection_buffer"
    # detection_buffer = DetectorBuffer(shm_name=detector_shm_name, is_owner=True)
    # detector_process = mp.Process(
    #     target=detector_runner, args=(camera_shm_name, detector_shm_name, stop_signal, frame_ready_signal)
    # )
    # detector_process.start()

    # try:
    #     while True:
    #         if camera_buffer.get_status():
    #             break
    #         time.sleep(1)
    #     while True:
    #         if detection_buffer.get_status():
    #             break
    #         time.sleep(1)

    #     while not stop_signal.is_set():
    #         pass

    # except KeyboardInterrupt:
    #     print("\n종료 신호 수신. 카메라 프로세스를 정리합니다...")
    #     stop_signal.set()

    # finally:
    #     print("자원 해제 및 프로세스 종료 중...")

    #     camera_process.join(timeout=3)
    #     if camera_process.is_alive():
    #         camera_process.terminate()

    #     detector_process.join(timeout=3)
    #     if detector_process.is_alive():
    #         detector_process.terminate()

    #     # control_process.join(timeout=3)
    #     # if control_process.is_alive():
    #     #     control_process.terminate()

    #     cv2.destroyAllWindows()
    #     camera_buffer.close()
    #     detection_buffer.close()
    #     print("모든 자원이 정상 해제되었습니다.")


if __name__ == "__main__":
    main()
