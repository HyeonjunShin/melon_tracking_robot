import time
import numpy as np
import multiprocessing as mp
from scipy.spatial.transform import Rotation as R

from lib.camera.gemini336 import Gemini336
from lib.camera.buffer import CameraBuffer
from lib.control.flange_buffer import FlangeBuffer
from lib.control.buffer import ControlBuffer

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


mouse_x, mouse_y = 0, 0


def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    mouse_x, mouse_y = x, y


def main():
    import cv2

    mp.set_start_method("spawn", force=True)
    stop_signal = mp.Event()
    frame_ready_signal = mp.Event()

    camera_shm_name = "camera_buffer"
    camera_buffer = CameraBuffer(shm_name=camera_shm_name, is_owner=True)
    camera_process = mp.Process(target=camera_runner, args=(camera_shm_name, stop_signal, frame_ready_signal))
    camera_process.start()

    flange_buffer = FlangeBuffer()

    cv2.namedWindow("color")
    cv2.setMouseCallback("color", mouse_callback)

    while True:
        if not frame_ready_signal.wait(timeout=0.035):
            continue
        frame = camera_buffer.get_latest_frame()
        frame_ready_signal.clear()

        Z = frame.depth[mouse_y][mouse_x].squeeze() * 0.001

        view = cv2.cvtColor(frame.color, cv2.COLOR_RGB2BGR)
        if Z > 0:
            u_distorted = float(mouse_x)
            v_distorted = float(mouse_y)
            point_pixel = np.array([u_distorted, v_distorted], dtype=np.float64)

            point_norm_camera = cv2.undistortPoints(point_pixel, K, D).squeeze()
            X, Y = point_norm_camera * Z

            point_3d_camera = np.array([X, Y, Z, 1], dtype=np.float32)
            point_3d_flange = TOOL_CAM @ point_3d_camera

            TF_FLANGE = flange_buffer.get_flange_tf(frame.timestamp)["T_base_flange"]
            TF_FLANGE[:3, 3] *= 0.001

            point_3d_robot = TF_FLANGE @ point_3d_flange
            print(frame.timestamp, mouse_x, mouse_y)
            print(X, Y, Z)
            print(point_3d_flange[0], point_3d_flange[1], point_3d_flange[2])
            print(point_3d_robot[0], point_3d_robot[1], point_3d_robot[2])

            # text = f"3D: X={X:.1f}mm, Y={Y:.1f}mm, Z={Z:.1f}mm"
            # cv2.putText(view, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.circle(view, (mouse_x, mouse_y), 5, (0, 0, 255), -1)

        cv2.imshow("color", view)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    stop_signal.set()
    camera_process.join(timeout=3)
    if camera_process.is_alive():
        camera_process.terminate()

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
