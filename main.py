import multiprocessing as mp
from lib.camera.gemini336 import LocklessBuffer, runner
import cv2
import time
import numpy as np


import torchvision.transforms.v2 as v2


def main():
    shm_name = "orbbec_frame_buffer"

    mp.set_start_method("spawn", force=True)
    stop_signal = mp.Event()
    camera_buffer = LocklessBuffer(shm_name=shm_name, is_owner=True)
    camera_process = mp.Process(target=runner, args=(shm_name, stop_signal))
    camera_process.start()

    prev_ts = None
    try:
        print("🚀 [Main Controller] 카메라 프로세스 실행 중 (종료하려면 'q' 또는 Ctrl+C)")
        while True:
            if camera_buffer.get_status():
                frame = camera_buffer.read_latest_frame()
                if prev_ts == frame.timestamp:
                    continue
                prev_ts = frame.timestamp
                
                # 1. Color 프레임 가져오기 (이미 BGR 포맷이라고 가정)
                color_img = frame.color
                color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
                
                # 2. Depth 프레임 전처리 (16비트 -> 8비트 시각화용 변환)
                depth_img = frame.depth
                # 0~5000mm(5m) 사이의 거리를 0~255 값으로 정규화 (카메라 스펙에 맞게 조절 가능)
                depth_clipped = np.clip(depth_img, 0, 5000)
                depth_normalized = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                # 깊이감을 보기 좋게 JET 컬러맵 적용 (가까운 곳은 빨간색/파란색 등)
                depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

                # 3. OpenCV 윈도우 표시
                cv2.imshow("Color Stream", color_img)
                cv2.imshow("Depth Stream", depth_colored)

                # 키 입력 처리 ('q' 누르면 안전 종료)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
    except KeyboardInterrupt:
        print("\n종료 신호 수신. 카메라 프로세스를 정리합니다...")
        
    finally:
        # 예외가 발생하더라도 자원이 확실히 해제되도록 보장
        print("자원 해제 및 프로세스 종료 중...")
        stop_signal.set()
        camera_process.join(timeout=3)
        if camera_process.is_alive():
            camera_process.terminate()

        # OpenCV 윈도우 닫기
        cv2.destroyAllWindows()
        camera_buffer.close()
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