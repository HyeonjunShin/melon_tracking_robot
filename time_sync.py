import cv2
import time
import numpy as np

np.set_printoptions(suppress=True)

import multiprocessing as mp
from utils import draw_3d_axis
import threading

from lib.camera.gemini336 import Gemini336
from lib.camera.buffer import CameraShm

from lib.tracker.tracker import CentroidTracker3D
from lib.control.ik_py import PyIk
from lib.control.dsr_py import DoosanRobotController

from lib.detector.detection import Detector, compute_pose, compute_pose_with_undistort
from lib.detector.detection import DetectorBuffer

from scipy.spatial.transform import Rotation as R

# TOOL_CAM = np.array(
#     [
#         [0.93969262, 0.00000000, 0.34202014, 0.05991575],
#         [0.00000000, 1.00000000, 0.00000000, 0.00358377],
#         [-0.34202014, 0.00000000, 0.93969262, 0.03416166],
#         [0.00000000, 0.00000000, 0.00000000, 1.00000000],
#     ],
#     dtype=np.float64,
# )

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

FX, FY = 693.3102, 693.4061
CX, CY = 639.6599, 365.0724
K = np.array(
    [
        [FX, 0.0, CX],
        [0.0, FY, CY],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
D = np.array(
    [
        0.00743896747007966,  # k1
        -0.05456198751926422,  # k2
        0.03670734167098999,  # p1 (또는 k3, 렌즈 모델에 따라 순서 확인 필요)
        0.0,  # p2
        0.0,  # k3
        0.0,  # k4
        0.00016195396892726421,  # k5
        -0.001005938509479165,  # k6
    ],
    dtype=np.float32,
)


def camera_runner(
    camera_shm_name,
    stop_signal,
    frame_ready_signal,
    color_shape=(1280, 720, 3),
    depth_shape=(1280, 720, 1),
):
    buffer = CameraShm(
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

        timestamp = color.get_global_timestamp_us()
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

    # def control_runner(detector_shm_name, stop_signal):
    #     # detector_buffer = DetectorBuffer(shm_name=detector_shm_name, is_owner=False)

    #     URDF = "/home/uon/Downloads/ik_solver-dev/data/robot/urdf/doosan_m1013.urdf"
    #     solver = PyIk(URDF)
    #     is_init_ik = False
    #     robot = DoosanRobotController("192.168.1.30", 500)

    #     INIT_JOINT = np.zeros((7,))
    #     TARGET_POSE = [0.6, 0.6, 0.5, 140, 180, -40]

    #     def ik_callback():
    #         d = 0.0005  # 1mm씩 움직임

    #         while True:
    #             if not is_init_ik:
    #                 continue

    #             # if TARGET_POSE[2] < 0.3 or TARGET_POSE[2] > 0.7:
    #             # d *= -1
    #             # TARGET_POSE[2] = TARGET_POSE[2] + d

    #             target_pose = TARGET_POSE  # TCP Pose

    #             target_matrix = PyIk.make_tf(
    #                 target_pose[0],
    #                 target_pose[1],
    #                 target_pose[2],  # x, y, z [m]
    #                 target_pose[3],
    #                 target_pose[4],
    #                 target_pose[5],  # roll, pitch, yaw [rad]
    #                 use_deg=False,
    #             )

    #             solver.movel(target_matrix)  # This Must be in threding
    #             time.sleep(0.001)

    #     robot.connect()
    #     time.sleep(0.1)
    #     robot.servo_on()
    #     time.sleep(3.0)
    #     robot.start_rt()

    #     if not solver.init():
    #         print("Error: PyIk 초기화 실패")
    #         return
    #     time.sleep(1.0)
    #     INIT_JOINT[:6] = robot.get_curr_joint_deg()
    #     # robot.movej(INIT_JOINT, 3.0)  # 3 sec moving
    #     solver.set_joint(INIT_JOINT, use_deg=True)
    #     # solver.set_tcp_max_speed(0.1)
    #     solver.set_end_effector_offset(T_FLANGE_CAMERA_Y_POS20)
    #     is_init_ik = True

    #     th = threading.Thread(target=ik_callback)
    #     th.start()

    #     while not stop_signal.is_set():
    #         if not detector_buffer.get_status():
    #             continue

    #         obj = detector_buffer.read_latest()
    #         if obj.detected:
    #             robot_ts, robot_tf = robot.get_flange_tf(obj.timestamp)

    #             curr_tf_meter = robot_tf.copy()
    #             curr_tf_meter[:3, 3] = curr_tf_meter[:3, 3] / 1000.0  # mm -> m

    #             R_base_flange = curr_tf_meter[:3, :3]  # Flange의 현재 회전 (3x3)
    #             P_base_flange = curr_tf_meter[:3, 3]  # Flange의 현재 위치 (m)

    #             T_flange_obj = np.eye(4, dtype=np.float64)
    #             T_flange_obj[:3, :3] = R.from_quat(obj.rotation).as_matrix()
    #             T_flange_obj[:3, 3] = obj.centroid

    #             # -------------------------------------------------------------
    #             # 3. Base 기준 개체 Pose 계산 (m 단위 통일 후 행렬 곱)
    #             # -------------------------------------------------------------
    #             T_base_obj = curr_tf_meter @ T_flange_obj

    #             # Base 기준 개체의 실제 3D 위치 [x, y, z] (m 단위)
    #             robot_coord = T_base_obj[:3, 3]

    #             target_quat = R.from_matrix(T_base_obj[:3, :3]).as_quat()
    #             rpy_deg = R.from_quat(target_quat).as_euler("zyz", degrees=True)

    #             # -------------------------------------------------------------
    #             # 4. Target Pose 업데이트
    #             # -------------------------------------------------------------
    #             TARGET_POSE[0] = robot_coord[0]  # m 단위 Base X
    #             TARGET_POSE[1] = robot_coord[1]  # m 단위 Base Y
    #             TARGET_POSE[2] = 0.6
    #             TARGET_POSE[3] = rpy_deg[0]  # Z1 (deg)
    #             TARGET_POSE[4] = rpy_deg[1]  # Y  (deg)
    #             TARGET_POSE[5] = rpy_deg[2]  # Z2 (deg)

    #             # print(robot_coord)

    #         res = solver.get_current_joint(use_deg=True)

    #         robot.movej_rt(res[0:6], 0.001)
    #         time.sleep(0.001)


def transform_cam_to_flange(centroid_cam, rotation_cam):
    T_cam_obj = np.eye(4, dtype=np.float64)

    T_cam_obj[:3, :3] = R.from_quat(rotation_cam).as_matrix()
    T_cam_obj[:3, 3] = centroid_cam

    T_flange_obj = TOOL_CAM @ T_cam_obj

    centroid_flange = T_flange_obj[:3, 3]
    rotation_flange = R.from_matrix(T_flange_obj[:3, :3]).as_quat()

    return centroid_flange, rotation_flange


def detector_runner(
    camera_shm_name,
    detector_shm_name,
    stop_signal,
    frame_ready_signal,
):
    camera_buffer = CameraShm(shm_name=camera_shm_name, is_owner=False)
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
            if not camera_buffer.get_status():
                detector_buffer.set_status(False)
                continue
            else:
                detector_buffer.set_status(True)

            if not frame_ready_signal.wait(timeout=0.035):
                print("Camera timeout!!")
                continue
            frame_ready_signal.clear()

            # loop_start = time.perf_counter()
            current_frame = camera_buffer.read()

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


def solver_runner(shm_name, stop_signal):
    URDF = "/home/uon/Downloads/ik_solver-dev/data/robot/urdf/doosan_m1013.urdf"
    solver = PyIk(URDF)
    is_init_ik = False

    while not stop_signal.is_set():
        if not is_init_ik:
            continue

        target_pose = TARGET_POSE  # TCP Pose

        target_matrix = PyIk.make_tf(
            target_pose[0],
            target_pose[1],
            target_pose[2],  # x, y, z [m]
            target_pose[3],
            target_pose[4],
            target_pose[5],  # roll, pitch, yaw [rad]
            use_deg=False,
        )

        solver.movel(target_matrix)  # This


u, v = 0, 0


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global u, v
        u, v = x, y


def main():
    mp.set_start_method("spawn", force=True)
    stop_signal = mp.Event()
    frame_ready_signal = mp.Event()

    camera_shm_name = "camera_buffer"
    camera_buffer = CameraShm(shm_name=camera_shm_name, is_owner=True)
    camera_process = mp.Process(target=camera_runner, args=(camera_shm_name, stop_signal, frame_ready_signal))
    camera_process.start()

    # detector_shm_name = "detection_buffer"
    # detector_buffer = DetectorBuffer(shm_name=detector_shm_name, is_owner=True)
    # detector_process = mp.Process(
    #     target=detector_runner, args=(camera_shm_name, detector_shm_name, stop_signal, frame_ready_signal)
    # )
    # detector_process.start()

    # control_process = mp.Process(target=control_runner, args=(detector_shm_name, stop_signal))
    # control_process.start()

    # URDF = "/home/uon/Downloads/ik_solver-dev/data/robot/urdf/doosan_m1013.urdf"
    # solver = PyIk(URDF)
    # is_init_ik = False
    robot = DoosanRobotController("192.168.1.30", 500)

    # INIT_JOINT = np.zeros((7,))
    # INIT_POSE = [-0.37411, 0.73885, 0.5, 180, 180, 90]
    # INIT_POSE = [0.480, 0.550, 0.6, 180, 180, 90]
    # INIT_POSE = [0.0, 0.9, 0.5, 180, 180, -90 + 90]
    # TARGET_POSE = INIT_POSE
    # TARGET_POSE = [0.0, 0.9, 0.5, 180, 180, -90 + 90]

    # def ik_callback():
    # # d = 0.0005  # 1mm씩 움직임

    # while True:
    #     if not is_init_ik:
    #         continue

    #     # if TARGET_POSE[2] < 0.3 or TARGET_POSE[2] > 0.7:
    #     # d *= -1
    #     # TARGET_POSE[2] = TARGET_POSE[2] + d

    #     target_pose = TARGET_POSE  # TCP Pose

    #     target_matrix = PyIk.make_tf(
    #         target_pose[0],
    #         target_pose[1],
    #         target_pose[2],  # x, y, z [m]
    #         target_pose[3],
    #         target_pose[4],
    #         target_pose[5],  # roll, pitch, yaw [rad]
    #         use_deg=False,
    #     )

    #     solver.movel(target_matrix)  # This Must be in threding
    # time.sleep(0.001)

    robot.connect()
    time.sleep(0.1)
    # robot.servo_on()
    # time.sleep(3.0)
    robot.start_rt()

    # if not solver.init():
    # print("Error: PyIk 초기화 실패")
    # return

    # time.sleep(1.0)
    # INIT_JOINT[:6] = robot.get_curr_joint_deg()
    # robot.movej(INIT_JOINT, 3.0)  # 3 sec moving
    # solver.set_joint(INIT_JOINT, use_deg=True)
    # solver.set_tcp_max_speed(1)
    # solver.set_end_effector_offset(TOOL_CAM)
    # is_init_ik = True

    # th = threading.Thread(target=ik_callback)
    # th.start()

    # cv2.namedWindow("color")
    # cv2.setMouseCallback("color", mouse_callback)
    try:
        print("🚀 [Main Controller] 카메라 및 비전 파이프라인 시각화 실행 중 ('q': 종료)")

        while True:
            if camera_buffer.get_status():
                break
            time.sleep(0.1)

        prev_ts = 0
        state = 0
        while True:
            frame = camera_buffer.read()
            ret = robot.get_flange_tf(frame.timestamp)
            # print(frame.timestamp )
            print(ret.target_ts, ret.d1_ts, ret.d2_ts)
            time.sleep(0.33)
            # color_img = cv2.cvtColor(frame.color, cv2.COLOR_RGB2BGR)

            # if state == 0:
            #     solver.set_end_effector_offset(TOOL_CAM)
            #     TARGET_POSE = INIT_POSE

            # if state == 1:
            #     _, current_tf = robot.get_flange_tf(time.time_ns())
            #     current_tf[:3, 3] *= 0.001  # m 변환
            #     current_tf = current_tf @ TOOL_SCUTION
            #     current_pos = current_tf[:3, 3]
            #     target_pos = np.array(TARGET_POSE[:3])
            #     err_pos = np.linalg.norm(target_pos - current_pos)

            #     target_rot_deg = np.array(TARGET_POSE[3:])
            #     target_rot = R.from_euler("ZYZ", target_rot_deg, degrees=True).as_matrix()
            #     current_rot = current_tf[:3, :3]
            #     R_diff = target_rot @ current_rot.T
            #     err_rot_vec = R.from_matrix(R_diff).as_rotvec(degrees=True)
            #     err_rot = np.linalg.norm(err_rot_vec)

            #     # print(f"위치 오차: {err_pos} mm | 회전 오차: {err_rot:.2f}°")
            #     if err_pos < 0.005 and err_rot < 1.0:
            #         state = 0

            # # 3. 화면 출력
            # cv2.imshow("color", color_img)

            # res = solver.get_current_joint(use_deg=True)
            # robot.movej_rt(res[0:6], 0.001)
            # time.sleep(0.001)

            # key = cv2.waitKey(1)
            # if key == ord("q"):
            #     break
            # if key == ord("c"):
            #     state = 0

            # if key == 32:
            #     solver.set_end_effector_offset(TOOL_SCUTION)

            #     depth_img = frame.depth
            #     _, current_tf = robot.get_flange_tf(frame.timestamp)

            #     patch_size = 10
            #     half_p = patch_size // 2

            #     h, w = depth_img.shape[:2]
            #     u_min, u_max = max(0, u - half_p), min(w, u + half_p)
            #     v_min, v_max = max(0, v - half_p), min(h, v + half_p)

            #     depth_roi = depth_img[v_min:v_max, u_min:u_max]

            #     valid_depths = depth_roi[depth_roi > 0]

            #     if len(valid_depths) > 0:
            #         z_mm = np.median(valid_depths)
            #     else:
            #         z_mm = depth_img[v][u][0]

            #     z = z_mm * 0.001  # mm -> m 변환
            #     # ------------------------------------------------------------------

            #     pixel_point = np.array([[[u, v]]], dtype=np.float32)
            #     undistorted_norm = cv2.undistortPoints(pixel_point, K, D)
            #     x_norm = undistorted_norm[0][0][0]
            #     y_norm = undistorted_norm[0][0][1]

            #     X_c = x_norm * z
            #     Y_c = y_norm * z
            #     Z_c = z

            #     camera_point_3d = np.array([X_c, Y_c, Z_c, 1], dtype=np.float32)
            #     # flange_point_3d = TOOL_CAM @ camera_point_3d

            #     # current_tf[:3, 3] = current_tf[:3, 3] * 0.001  # mm to m
            #     # base_point_3d = current_tf @ flange_point_3d

            #     current_tf[:3, 3] *= 0.001  # mm to m
            #     T_base_cam = current_tf @ TOOL_CAM
            #     base_point_3d = T_base_cam @ camera_point_3d

            #     print(base_point_3d)

            #     TARGET_POSE = [
            #         base_point_3d[0],
            #         base_point_3d[1],
            #         base_point_3d[2],
            #         180.0,
            #         180.0,
            #         90.0,
            #     ]
            #     state = 1

    except KeyboardInterrupt:
        print("\n종료 신호 수신. 카메라 프로세스를 정리합니다...")
        stop_signal.set()

    finally:
        print("자원 해제 및 프로세스 종료 중...")

        camera_process.join(timeout=3)
        if camera_process.is_alive():
            camera_process.terminate()

        # detector_process.join(timeout=3)
        # if detector_process.is_alive():
        #     detector_process.terminate()

        # control_process.join(timeout=3)
        # if control_process.is_alive():
        #     control_process.terminate()

        # cv2.destroyAllWindows()
        # camera_buffer.close()
        # detector_buffer.close()
        # print("모든 자원이 정상 해제되었습니다.")

    # prev_ts = 0.0
    # try:
    #     while True:
    #         current_frame = buffer.read_latest_frame()
    #         # # 동일한 타임스탬프 프레임이면 추론 스킵 (Polling 대기)
    #         # if current_frame.timestamp == prev_ts or current_frame.timestamp == 0.0:
    #             # time.sleep(0.002)
    #             # continue
    #         print(current_frame.timestamp)import time


# import numpy as np
# import multiprocessing as mp
# from multiprocessing import shared_memory
# from pathlib import Path
# from typing import Optional, Tuple

# from pyorbbecsdk import (
#     Pipeline, Config, Context, AlignFilter,
#     OBSensorType, OBFormat, OBPropertyID, OBStreamType
# )

# # --- Configuration Constant ---
# CURRENT_DIR = Path(__file__).resolve().parent
# PRESET_JSON_PATH = (CURRENT_DIR / "gemini336_settings.json").as_posix()

# COLOR_WIDTH, COLOR_HEIGHT, COLOR_FPS = 1280, 720, 30
# DEPTH_WIDTH, DEPTH_HEIGHT, DEPTH_FPS = 1280, 720, 30
# SHM_NAME = "orbbec_frame_buffer"

# TIMESTAMP_BYTES = int(np.uint64().itemsize)
# COLOR_BYTES = int(np.prod((COLOR_HEIGHT, COLOR_WIDTH, 3)) * np.uint8().itemsize)
# DEPTH_BYTES = int(np.prod((DEPTH_HEIGHT, DEPTH_WIDTH, 1)) * np.uint16().itemsize)
# FRAME_BYTES = TIMESTAMP_BYTES + COLOR_BYTES + DEPTH_BYTES


# class Frame:
#     def __init__(self, timestamp: int, color: np.ndarray, depth: np.ndarray):
#         self.timestamp = timestamp
#         self.color = color
#         self.depth = depth

#     def __repr__(self):
#         return f"<Frame TS:{self.timestamp:.2f} | Color:{self.color.shape} | Depth:{self.depth.shape}>"


# class LocklessBuffer:
#     """공유 메모리 기반 Double Buffer Wrapper"""
#     def __init__(self, name: str = SHM_NAME, is_owner: bool = False):
#         self.shm_name = name
#         self.is_owner = is_owner
#         self.header_bytes = 16
#         self.total_bytes = self.header_bytes + (FRAME_BYTES * 2)

#         if self.is_owner:
#             try:
#                 old_shm = shared_memory.SharedMemory(name=self.shm_name)
#                 old_shm.close()
#                 old_shm.unlink()
#             except FileNotFoundError:
#                 pass
#             self.shm = shared_memory.SharedMemory(name=self.shm_name, create=True, size=self.total_bytes)
#         else:
#             self.shm = shared_memory.SharedMemory(name=self.shm_name, create=False)

#         self.status_arr = np.ndarray((1,), dtype=np.bool_, buffer=self.shm.buf, offset=0)
#         self.write_index_arr = np.ndarray((1,), dtype=np.int64, buffer=self.shm.buf, offset=8)

#         if self.is_owner:
#             self.status_arr[0] = False
#             self.write_index_arr[0] = 0

#         self.slots = []
#         for i in range(2):
#             slot_offset = self.header_bytes + (i * FRAME_BYTES)
#             ts_offset = slot_offset
#             color_offset = slot_offset + TIMESTAMP_BYTES
#             depth_offset = slot_offset + TIMESTAMP_BYTES + COLOR_BYTES

#             ts_arr = np.ndarray((1,), dtype=np.uint64, buffer=self.shm.buf, offset=ts_offset)
#             color_arr = np.ndarray((COLOR_HEIGHT, COLOR_WIDTH, 3), dtype=np.uint8, buffer=self.shm.buf, offset=color_offset)
#             depth_arr = np.ndarray((DEPTH_HEIGHT, DEPTH_WIDTH, 1), dtype=np.uint16, buffer=self.shm.buf, offset=depth_offset)
#             self.slots.append({"ts": ts_arr, "color": color_arr, "depth": depth_arr})

#     def write(self, timestamp: float, color_data: np.ndarray, depth_data: np.ndarray):
#         current_idx = int(self.write_index_arr[0])
#         next_idx = 1 - current_idx

#         target_slot = self.slots[next_idx]
#         target_slot["ts"][0] = timestamp

#         if color_data is not None:
#             np.copyto(target_slot["color"], color_data.reshape((COLOR_HEIGHT, COLOR_WIDTH, 3)))

#         if depth_data is not None:
#             depth_data_u16 = depth_data.view(np.uint16)
#             np.copyto(target_slot["depth"], depth_data_u16.reshape((DEPTH_HEIGHT, DEPTH_WIDTH, 1)))

#         self.write_index_arr[0] = next_idx

#     def read_latest_frame(self) -> Frame:
#         latest_idx = int(self.write_index_arr[0])
#         slot = self.slots[latest_idx]

#         return Frame(
#             timestamp=int(slot["ts"][0]),
#             color=slot["color"].copy(),
#             depth=slot["depth"].copy(),
#         )

#     def get_status(self) -> bool:
#         return bool(self.status_arr[0])

#     def set_status(self, is_good: bool):
#         self.status_arr[0] = is_good

#     def close(self):
#         self.shm.close()
#         if self.is_owner:
#             try:
#                 self.shm.unlink()
#             except FileNotFoundError:
#                 pass


# class OrbbecCamera:
#     def __init__(self, device):
#         self.device = device
#         self.info = device.get_device_info()
#         self.serial_number = self.info.get_serial_number()
#         self.device_name = self.info.get_name()
#         self.pipeline: Optional[Pipeline] = None
#         self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

#     def start(self):
#         try:
#             self.device.load_preset_from_json_file(PRESET_JSON_PATH)
#         except Exception as e:
#             print(f"⚠️ 프리셋 로드 실패 ({self.serial_number}): {e}")

#         self.pipeline = Pipeline(self.device)
#         color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
#         color_profile = color_profiles.get_video_stream_profile(COLOR_WIDTH, COLOR_HEIGHT, OBFormat.RGB, COLOR_FPS)

#         depth_profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
#         depth_profile = depth_profiles.get_video_stream_profile(DEPTH_WIDTH, DEPTH_HEIGHT, OBFormat.Y16, DEPTH_FPS)

#         config = Config()
#         config.enable_stream(color_profile)
#         config.enable_stream(depth_profile)

#         self.pipeline.start(config)
#         self.print_intrinsics()

#     def get_aligned_frame(self, timeout_ms: int = 100) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
#         if not self.pipeline:
#             return None

#         frames = self.pipeline.wait_for_frames(timeout_ms)
#         if frames is None:
#             return None

#         aligned_frames = self.align_filter.process(frames)
#         if not aligned_frames:
#             return None

#         color_frame = aligned_frames.get_color_frame()
#         depth_frame = aligned_frames.get_depth_frame()

#         if not color_frame or not depth_frame:
#             return None

#         ts = depth_frame.get_global_timestamp_us()
#         return ts, color_frame.get_data(), depth_frame.get_data()

#     def stop(self):
#         if self.pipeline:
#             try:
#                 self.pipeline.stop()
#             except Exception:
#                 pass
#             self.pipeline = None

#     def print_intrinsics(self):
#         if not self.pipeline:
#             return
#         try:
#             param = self.pipeline.get_camera_param()
#             c_intrin = param.rgb_intrinsic
#             print(f"\n[Color Camera Intrinsic] Res: {c_intrin.width}x{c_intrin.height}, fx/fy: ({c_intrin.fx:.2f}, {c_intrin.fy:.2f})")
#         except Exception as e:
#             print(f"🛑 Intrinsic 조회 실패: {e}")


# class CameraRunner:
#     """카메라 프로세스 루프 및 이벤트 핫플러깅 관리 클래스"""
#     def __init__(self, shm_name: str):
#         self.shm_name = shm_name
#         self.buffer: Optional[LocklessBuffer] = None
#         self.camera: Optional[OrbbecCamera] = None
#         self.ctx: Optional[Context] = None

#     def _on_device_changed(self, removed_list, added_list):
#         if removed_list.get_count() > 0:
#             print("🛑 카메라 제거됨")
#             if self.camera:
#                 self.camera.stop()
#                 self.camera = None
#                 self.buffer.set_status(False)

#         if added_list.get_count() > 0:
#             print("▶ 카메라 감지됨:")
#             device = added_list.get_device_by_index(0)
#             self._init_camera(device)

#     def _init_camera(self, device):
#         try:
#             cam = OrbbecCamera(device)
#             cam.start()
#             self.camera = cam
#             print(f"✅ 카메라 연결 성공: {cam.serial_number}")
#         except Exception as e:
#             print(f"❌ 카메라 초기화 실패: {e}")

#     def run(self, stop_signal: mp.Event):
#         print("🎥 [Camera Runner] 실행 시작")
#         self.buffer = LocklessBuffer(name=self.shm_name, is_owner=False)

#         self.ctx = Context()
#         self.ctx.set_device_changed_callback(self._on_device_changed)

#         device_list = self.ctx.query_devices()
#         if device_list.get_count() > 0:
#             self._init_camera(device_list.get_device_by_index(0))

#         try:
#             while not stop_signal.is_set():
#                 if self.camera is None:
#                     time.sleep(0.01)
#                     continue

#                 try:
#                     frame_data = self.camera.get_aligned_frame(timeout_ms=100)
#                     if frame_data is None:
#                         continue

#                     ts, color, depth = frame_data
#                     self.buffer.write(ts, color, depth)

#                     if not self.buffer.get_status():
#                         self.buffer.set_status(True)

#                 except Exception as e:
#                     print(f"⚠️ 프레임 수신 중 에러: {e}")
#                     if self.camera:
#                         self.camera.stop()
#                         self.camera = None

#         except KeyboardInterrupt:
#             pass

#         # Clean up
#         if self.camera:
#             self.camera.stop()
#         self.buffer.close()
#         del self.ctx
#         print("🎥 [Camera Runner] 프로세스 정상 종료")


# def run_camera_process(shm_name: str, stop_signal: mp.Event):
#     runner = CameraRunner(shm_name)
#     runner.run(stop_signal)


# if __name__ == "__main__":
#     mp.set_start_method("spawn", force=True)

#     buffer = LocklessBuffer(name=SHM_NAME, is_owner=True)
#     stop_signal = mp.Event()

#     cam_process = mp.Process(target=run_camera_process, args=(SHM_NAME, stop_signal))
#     cam_process.start()

#     try:
#         print("🚀 [Main Controller] 실행 중 (Ctrl+C 종료)")
#         while True:
#             if buffer.get_status():
#                 frame = buffer.read_latest_frame()
#                 print(f"Frame TS: {frame.timestamp}")
#             time.sleep(0.03)  # Loop delay control
#     except KeyboardInterrupt:
#         print("\n종료 신호 수신. 카메라 프로세스 정리 중...")
#         stop_signal.set()
#         cam_process.join(timeout=3)
#         if cam_process.is_alive():
#             cam_process.terminate()

#         buffer.close()
#         print("모든 자원 정리 완료.")

#         # prev_ts = current_frame.timestamp

#         # color = current_frame.color
#         # depth = current_frame.depth
#         # view = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

#         # input_tensor = transform(color)[None, ...]
#         # input_tensor = input_tensor.to(device)

#         # with torch.inference_mode():
#         #     cls_pred, reg_pred = model(input_tensor)

#         #     decoded_bboxes = decoder.decode(reg_pred)[0]
#         #     pred_scores = cls_pred[0].sigmoid().squeeze(-1)

#         #     keep = pred_scores > CONF_THRES
#         #     final_boxes = decoded_bboxes[keep]
#         #     final_scores = pred_scores[keep]

#         # measured_3d = None
#         # if len(final_boxes) > 0:
#         #     final_boxes = final_boxes.clone()
#         #     final_boxes[:, [0, 2]] = final_boxes[:, [0, 2]] * 2
#         #     final_boxes[:, [1, 3]] = (final_boxes[:, [1, 3]] - 12) * 2

#         #     best_idx = torch.argmax(final_scores)
#         #     best_box = final_boxes[best_idx].tolist()

#         #     # 3D Centroid (Xc, Yc, Zc) 측정
#         #     measured_3d = kf_tracker.compute_3d_centroid(
#         #         depth, best_box, FX, FY, CX, CY
#         #     )

#         #     x1, y1, x2, y2 = map(int, best_box)
#         #     cv2.rectangle(view, (x1, y1), (x2, y2), (80, 80, 80), 1)

#         # state, is_updated = kf_tracker.update(measured_3d)

#         # if kf_tracker.is_initialized:
#         #     xc, yc, zc, vx, vy, vz = state

#         #     # 3D TF 축 시각화
#         #     draw_3d_tf_axis(
#         #         img=view,
#         #         center_3d=(xc, yc, zc),
#         #         fx=FX,
#         #         fy=FY,
#         #         cx=CX,
#         #         cy=CY,
#         #         axis_length=80.0,
#         #         thickness=2,
#         #     )

#         #     # 3D Pos & Speed 텍스트
#         #     pos_text = f"TF [X:{xc:.0f}, Y:{yc:.0f}, Z:{zc:.0f}] mm"
#         #     vel_text = f"V [Vx:{vx:.0f}, Vy:{vy:.0f}, Vz:{vz:.0f}] mm/s"
#         #     cv2.putText(
#         #         view,
#         #         pos_text,
#         #         (10, 30),
#         #         cv2.FONT_HERSHEY_SIMPLEX,
#         #         0.6,
#         #         (0, 255, 255),
#         #         2,
#         #         cv2.LINE_AA,
#         #     )
#         #     cv2.putText(
#         #         view,
#         #         vel_text,
#         #         (10, 60),
#         #         cv2.FONT_HERSHEY_SIMPLEX,
#         #         0.6,
#         #         (255, 255, 0),
#         #         2,
#         #         cv2.LINE_AA,
#         #     )

#         # curr_time = time.time()
#         # fps = 0.9 * fps + 0.1 * (1.0 / (curr_time - prev_time + 1e-6))
#         # prev_time = curr_time
#         # cv2.putText(
#         #     view,
#         #     f"FPS: {fps:.1f}",
#         #     (15, 90),
#         #     cv2.FONT_HERSHEY_SIMPLEX,
#         #     0.8,
#         #     (0, 0, 255),
#         #     2,
#         #     cv2.LINE_AA,
#         # )
#         # cv2.imshow("color", view)
#         # key = cv2.waitKey(1)

#         # if key == ord("q"):
#         #     break

# finally:
#     # cv2.destroyAllWindows()
#     stop_signal.set()
#     camera_process.join(timeout=3)
#     if camera_process.is_alive():
#         camera_process.terminate()
#     buffer.close()
#     print("[메인] 시스템이 안전하게 종료되었습니다.")


if __name__ == "__main__":
    main()
