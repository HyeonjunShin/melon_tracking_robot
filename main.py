import cv2
import time
import numpy as np
import multiprocessing as mp
from utils import draw_3d_axis
import threading

from lib.tracker.tracker import CentroidTracker3D
from lib.control.ik_py import PyIk
from lib.control.dsr_py import DoosanRobotController

from scipy.spatial.transform import Rotation as R

T_FLANGE_CAMERA_Y_POS20 = np.array(
    [
        [0.93969262, 0.00000000, 0.34202014, 0.05991575],
        [0.00000000, 1.00000000, 0.00000000, 0.00358377],
        [-0.34202014, 0.00000000, 0.93969262, 0.03416166],
        [0.00000000, 0.00000000, 0.00000000, 1.00000000],
    ],
    dtype=np.float64,
)

def control_runner(detector_shm_name, stop_signal):
    detector_buffer = DetectorBuffer(shm_name=detector_shm_name, is_owner=False)

    URDF = "/home/uon/Downloads/ik_solver-dev/data/robot/urdf/doosan_m1013.urdf"
    solver = PyIk(URDF)
    is_init_ik = False
    robot = DoosanRobotController("192.168.1.30", 500)

    INIT_JOINT = np.zeros((7,))
    TARGET_POSE = [0.6, 0.6, 0.5, 140, 180, -40]

    def ik_callback():
        d = 0.0005  # 1mm씩 움직임

        while True:
            if not is_init_ik:
                continue

            # if TARGET_POSE[2] < 0.3 or TARGET_POSE[2] > 0.7:
            # d *= -1
            # TARGET_POSE[2] = TARGET_POSE[2] + d

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

            solver.movel(target_matrix)  # This Must be in threding
            time.sleep(0.001)

    robot.connect()
    time.sleep(0.1)
    robot.servo_on()
    time.sleep(3.0)
    robot.start_rt()

    if not solver.init():
        print("Error: PyIk 초기화 실패")
        return
    time.sleep(1.0)
    INIT_JOINT[:6] = robot.get_curr_joint_deg()
    # robot.movej(INIT_JOINT, 3.0)  # 3 sec moving
    solver.set_joint(INIT_JOINT, use_deg=True)
    # solver.set_tcp_max_speed(0.1)
    solver.set_end_effector_offset(T_FLANGE_CAMERA_Y_POS20)
    is_init_ik = True

    th = threading.Thread(target=ik_callback)
    th.start()

    while not stop_signal.is_set():
        if not detector_buffer.get_status():
            continue

        obj = detector_buffer.read_latest()
        if obj.detected:
            robot_ts, robot_tf = robot.get_flange_tf(obj.timestamp)

            curr_tf_meter = robot_tf.copy()
            curr_tf_meter[:3, 3] = curr_tf_meter[:3, 3] / 1000.0  # mm -> m

            R_base_flange = curr_tf_meter[:3, :3]  # Flange의 현재 회전 (3x3)
            P_base_flange = curr_tf_meter[:3, 3]  # Flange의 현재 위치 (m)

            T_flange_obj = np.eye(4, dtype=np.float64)
            T_flange_obj[:3, :3] = R.from_quat(obj.rotation).as_matrix()
            T_flange_obj[:3, 3] = obj.centroid

            # -------------------------------------------------------------
            # 3. Base 기준 개체 Pose 계산 (m 단위 통일 후 행렬 곱)
            # -------------------------------------------------------------
            T_base_obj = curr_tf_meter @ T_flange_obj

            # Base 기준 개체의 실제 3D 위치 [x, y, z] (m 단위)
            robot_coord = T_base_obj[:3, 3]

            target_quat = R.from_matrix(T_base_obj[:3, :3]).as_quat()
            rpy_deg = R.from_quat(target_quat).as_euler("zyz", degrees=True)

            # -------------------------------------------------------------
            # 4. Target Pose 업데이트
            # -------------------------------------------------------------
            TARGET_POSE[0] = robot_coord[0]  # m 단위 Base X
            TARGET_POSE[1] = robot_coord[1]  # m 단위 Base Y
            TARGET_POSE[2] = 0.6
            TARGET_POSE[3] = rpy_deg[0]  # Z1 (deg)
            TARGET_POSE[4] = rpy_deg[1]  # Y  (deg)
            TARGET_POSE[5] = rpy_deg[2]  # Z2 (deg)

            # print(robot_coord)

        res = solver.get_current_joint(use_deg=True)

        robot.movej_rt(res[0:6], 0.001)
        time.sleep(0.001)


def main():
    mp.set_start_method("spawn", force=True)
    stop_signal = mp.Event()
    frame_ready_signal = mp.Event()


    detector_shm_name = "detection_buffer"
    detector_buffer = DetectorBuffer(shm_name=detector_shm_name, is_owner=True)
    detector_process = mp.Process(
        target=detector_runner, args=(camera_shm_name, detector_shm_name, stop_signal, frame_ready_signal)
    )
    detector_process.start()

    control_process = mp.Process(target=control_runner, args=(detector_shm_name, stop_signal))
    control_process.start()

    try:
        print("🚀 [Main Controller] 카메라 및 비전 파이프라인 시각화 실행 중 ('q': 종료)")

        FX, FY = 693.3102, 693.4061
        CX, CY = 639.6599, 365.0724

        prev_ts = 0

        while True:
            if camera_buffer.get_status():
                frame = camera_buffer.read_latest_frame()
                if prev_ts == frame.timestamp:
                    time.sleep(0.001)  # CPU Overhead 방지
                    continue
                prev_ts = frame.timestamp
            else:
                continue
            color_img = cv2.cvtColor(frame.color, cv2.COLOR_RGB2BGR)

            if detector_buffer.get_status():
                detector_ret = detector_buffer.read_latest()

                centroid = detector_ret.centroid  # [X, Y, Z] (m)
                # velocity = detector_ret.velocity  # [Vx, Vy, Vz] (m/s)
                rotation = detector_ret.rotation

                x, y, z = centroid
                # vx, vy, vz = velocity
                # speed = np.linalg.norm(velocity)  # 속도 크기 스칼라 (m/s)

                # ----------------------------------------------------------------------
                # 1. 디텍터가 실제로 객체를 감지한 경우 (Active Detection)
                # ----------------------------------------------------------------------
                if detector_ret.detected:
                    x1, y1, x2, y2 = map(int, detector_ret.bbox)
                    score = detector_ret.score

                    # A. Bounding Box 시각화 (초록색 상자)
                    cv2.rectangle(color_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # B. ROI 중앙 60% 영역 표시
                    w, h = x2 - x1, y2 - y1
                    scale = np.sqrt(0.6)
                    mw, mh = int((w * (1 - scale)) / 2), int((h * (1 - scale)) / 2)
                    cv2.rectangle(
                        color_img,
                        (x1 + mw, y1 + mh),
                        (x2 - mw, y2 - mh),
                        (0, 255, 255),
                        1,
                    )

                    # C. BBox 상단 라벨 (Score 및 상태)
                    info_text = f"DETECTED | Score: {score:.2f}"
                    (text_w, text_h), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(
                        color_img,
                        (x1, max(0, y1 - 22)),
                        (x1 + text_w, y1),
                        (0, 255, 0),
                        -1,
                    )
                    cv2.putText(
                        color_img,
                        info_text,
                        (x1, max(12, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )

                    # D. 화면 좌상단 3D 위치 및 속도 종합 오버레이 패널
                    pos_text = f"[3D POS]  X: {x:+.2f}m | Y: {y:+.2f}m | Z: {z:+.2f}m"
                    # vel_text = f"[3D VEL]  Vx: {vx:+.2f}m/s | Vy: {vy:+.2f}m/s | Vz: {vz:+.2f}m/s (Speed: {speed:.2f}m/s)"

                    # 배경 검은색 반투명 박스 (가독성 향상)
                    cv2.rectangle(color_img, (10, 10), (620, 75), (0, 0, 0), -1)
                    cv2.rectangle(color_img, (10, 10), (620, 75), (0, 255, 0), 1)

                    cv2.putText(
                        color_img,
                        pos_text,
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                    # cv2.putText(
                    #     color_img,
                    #     vel_text,
                    #     (20, 60),
                    #     cv2.FONT_HERSHEY_SIMPLEX,
                    #     0.55,
                    #     (0, 255, 255),
                    #     1,
                    #     cv2.LINE_AA,
                    # )

                    # E. 3D Orientation Axis 투영
                    draw_3d_axis(
                        color_img,
                        centroid=centroid,
                        rotation=rotation,
                        fx=FX,
                        fy=FY,
                        cx=CX,
                        cy=CY,
                        axis_length=0.1,
                    )

                # ----------------------------------------------------------------------
                # 2. 디텍션은 놓쳤으나, 칼만 필터가 추적(Predict) 중인 경우
                # ----------------------------------------------------------------------
                elif not np.all(centroid == 0):
                    pos_text = f"[PREDICT POS] X: {x:+.2f}m | Y: {y:+.2f}m | Z: {z:+.2f}m"
                    vel_text = f"[PREDICT VEL] Vx: {vx:+.2f}m/s | Vy: {vy:+.2f}m/s | Vz: {vz:+.2f}m/s (Speed: {speed:.2f}m/s)"

                    # 주황색 경고 테두리 패널
                    cv2.rectangle(color_img, (10, 10), (650, 75), (0, 0, 0), -1)
                    cv2.rectangle(color_img, (10, 10), (650, 75), (0, 165, 255), 1)

                    cv2.putText(
                        color_img,
                        pos_text,
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 165, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        color_img,
                        vel_text,
                        (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                    # 이전 Bounding Box 주황색 표시
                    if not np.all(detector_ret.bbox == 0):
                        x1, y1, x2, y2 = map(int, detector_ret.bbox)
                        cv2.rectangle(color_img, (x1, y1), (x2, y2), (0, 165, 255), 1)

                    # 예측 위치에 3D 좌표축 투영
                    draw_3d_axis(
                        color_img,
                        centroid=centroid,
                        rotation=rotation,
                        fx=FX,
                        fy=FY,
                        cx=CX,
                        cy=CY,
                        axis_length=0.1,
                    )

                # ----------------------------------------------------------------------
                # 3. 아예 추적 상태가 아닌 경우 (초기화 전 또는 미인식)
                # ----------------------------------------------------------------------
                else:
                    cv2.rectangle(color_img, (10, 10), (220, 45), (0, 0, 0), -1)
                    cv2.putText(
                        color_img,
                        "SEARCHING...",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

            # 3. 화면 출력
            cv2.imshow("Color Stream (BBox & 3D Pose)", color_img)

            # 'q' 키 입력 시 안전 종료
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n종료 신호 수신. 카메라 프로세스를 정리합니다...")
        stop_signal.set()

    finally:
        print("자원 해제 및 프로세스 종료 중...")

        camera_process.join(timeout=3)
        if camera_process.is_alive():
            camera_process.terminate()

        detector_process.join(timeout=3)
        if detector_process.is_alive():
            detector_process.terminate()

        control_process.join(timeout=3)
        if control_process.is_alive():
            control_process.terminate()

        cv2.destroyAllWindows()
        camera_buffer.close()
        detector_buffer.close()
        print("모든 자원이 정상 해제되었습니다.")

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
