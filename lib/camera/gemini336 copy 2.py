import time
import numpy as np
import multiprocessing as mp
from pathlib import Path
from typing import Optional, Tuple, Dict

from pyorbbecsdk import (
    Pipeline,
    Config,
    Context,
    AlignFilter,
    OBSensorType,
    OBFormat,
    OBPropertyID,
    OBStreamType,
)

# from lib.camera.camera_shm import CameraBuffer
from lib.camera.buffer import CameraShm

CHECK_PARAMS = [
    (
        "color_auto_exposure",
        OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL,
        "bool",
    ),
    (
        "color_auto_white_balance",
        OBPropertyID.OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL,
        "bool",
    ),
    (
        "color_backlight_compensation",
        OBPropertyID.OB_PROP_COLOR_BACKLIGHT_COMPENSATION_INT,
        "int",
    ),
    ("color_brightness", OBPropertyID.OB_PROP_COLOR_BRIGHTNESS_INT, "int"),
    ("color_contrast", OBPropertyID.OB_PROP_COLOR_CONTRAST_INT, "int"),
    ("color_exposure_time", OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT, "int"),
    ("color_gain", OBPropertyID.OB_PROP_COLOR_GAIN_INT, "int"),
    ("color_gamma", OBPropertyID.OB_PROP_COLOR_GAMMA_INT, "int"),
    ("color_hue", OBPropertyID.OB_PROP_COLOR_HUE_INT, "int"),
    (
        "color_power_line_frequency",
        OBPropertyID.OB_PROP_COLOR_POWER_LINE_FREQUENCY_INT,
        "int",
    ),
    ("color_saturation", OBPropertyID.OB_PROP_COLOR_SATURATION_INT, "int"),
    ("color_sharpness", OBPropertyID.OB_PROP_COLOR_SHARPNESS_INT, "int"),
    (
        "color_white_balance",
        OBPropertyID.OB_PROP_COLOR_WHITE_BALANCE_INT,
        "int",
    ),
    (
        "depth_auto_exposure",
        OBPropertyID.OB_PROP_DEPTH_AUTO_EXPOSURE_BOOL,
        "bool",
    ),
    ("depth_exposure_time", OBPropertyID.OB_PROP_DEPTH_EXPOSURE_INT, "int"),
    ("depth_gain", OBPropertyID.OB_PROP_DEPTH_GAIN_INT, "int"),
    ("laser_state", OBPropertyID.OB_PROP_LASER_CONTROL_INT, "int"),
    (
        "laser_power_level",
        OBPropertyID.OB_PROP_LASER_POWER_LEVEL_CONTROL_INT,
        "int",
    ),
    ("HW_align", OBPropertyID.OB_PROP_DEPTH_ALIGN_HARDWARE_BOOL, "bool"),
]


class OrbbecCamera:
    def __init__(self, device, color_shape, depth_shape, settings_path):
        self.color_shape = color_shape
        self.depth_shape = depth_shape
        self.settings_path = settings_path

        self.device = device
        self.info = device.get_device_info()
        self.serial_number = self.info.get_serial_number()
        self.pipeline: Optional[Pipeline] = None

    def check_parameters(self):
        for label, prop_id, prop_type in CHECK_PARAMS:
            try:
                if prop_type == "bool":
                    val = self.device.get_bool_property(prop_id)
                elif prop_type == "float":
                    val = self.device.get_float_property(prop_id)
                else:
                    val = self.device.get_int_property(prop_id)
                print(f" ▶ {label:<30} : {val}")
            except Exception:
                print(f" 🛑 {label:<30} : [조회 실패 / 지원하지 않는 프로퍼티]")

    def print_intrinsics(self):
        if not self.pipeline:
            return
        try:
            param = self.pipeline.get_camera_param()

            c_intrin = param.rgb_intrinsic
            print("\n" + "=" * 50)
            print(" [Color Camera Intrinsic Parameters]")
            print(f"  ▶ Resolution : {c_intrin.width} x {c_intrin.height}")
            print(f"  ▶ Focal Length (fx, fy) : ({c_intrin.fx:.4f}, {c_intrin.fy:.4f})")
            print(f"  ▶ Principal Point (cx, cy) : ({c_intrin.cx:.4f}, {c_intrin.cy:.4f})")
            c_dist = param.rgb_distortion
            print(
                f"  ▶ Distortion Model : {c_dist.k1} {c_dist.k2} {c_dist.k3} {c_dist.k4} {c_dist.k5} {c_dist.k6} {c_dist.p1} {c_dist.p2}"
            )

            d_intrin = param.depth_intrinsic
            print("-" * 50)
            print(" [Depth Camera Intrinsic Parameters]")
            print(f"  ▶ Resolution : {d_intrin.width} x {d_intrin.height}")
            print(f"  ▶ Focal Length (fx, fy) : ({d_intrin.fx:.4f}, {d_intrin.fy:.4f})")
            print(f"  ▶ Principal Point (cx, cy) : ({d_intrin.cx:.4f}, {d_intrin.cy:.4f})")
            d_dist = param.depth_distortion
            print(
                f"  ▶ Distortion Model : {d_dist.k1} {d_dist.k2} {d_dist.k3} {d_dist.k4} {d_dist.k5} {d_dist.k6} {d_dist.p1} {d_dist.p2}"
            )
            print("=" * 50 + "\n")

        except Exception as e:
            print(f" 🛑 내적 파라미터(Intrinsic) 조회 실패: {e}")

    def start(self):
        try:
            self.device.load_preset_from_json_file(self.settings_path)
            print(f"✅ 프리셋 로드 성공: {self.settings_path}")
        except Exception as e:
            print(f"프리셋 로드 실패 ({self.serial_number}): {e}")

        # [순서 핵심 2] Pipeline 생성 및 스트림 프로필 설정
        self.pipeline = Pipeline(self.device)
        color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = color_profiles.get_video_stream_profile(
            self.color_shape[0], self.color_shape[1], OBFormat.RGB, 30
        )

        depth_profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = depth_profiles.get_video_stream_profile(
            self.depth_shape[0], self.depth_shape[1], OBFormat.Y16, 30
        )

        config = Config()
        config.enable_stream(color_profile)
        config.enable_stream(depth_profile)

        # [순서 핵심 3] Pipeline Start
        self.pipeline.start(config)

        # [순서 핵심 4] Intrinsic 및 Parameter 출력
        self.print_intrinsics()
        self.check_parameters()

    def stop(self):
        if self.pipeline:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None


class CameraManager:
    def __init__(self, buffer: CameraShm):
        current_dir = Path(__file__).resolve().parent
        self.settings_path = str((current_dir / "gemini336_settings.json").resolve())
        self.shm_name = shm_name
        self.camera = None
        self.buffer = buffer
        self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

    def _on_device_changed(self, removed_list, added_list):
        if removed_list.get_count() > 0:
            print("Deleted the camera")
            for i in range(removed_list.get_count()):
                serial_number = removed_list.get_device_serial_number_by_index(i)
                cam = self.cameras.pop(serial_number, None)
                if cam:
                    cam.stop()
                    del cam
                print(f"  - {removed_list.get_device_name_by_index(i)} (SN: {serial_number})")
                self.buffer.set_status(False)

        if added_list.get_count() > 0:
            print("Added the camera:")
            for i in range(added_list.get_count()):
                device = added_list.get_device_by_index(i)
                info = device.get_device_info()
                serial_number = info.get_serial_number()
                device_name = info.get_name()

                print(f"[{i}] {device_name} (SN: {serial_number})")
                try:

                    cam = OrbbecCamera(
                        device=device,
                        color_shape=(1280, 720, 3),
                        depth_shape=(1280, 720, 1),
                        settings_path=self.settings_path,
                    )
                    cam.start()
                    self.cameras[serial_number] = cam
                    print(f"[Success connection] {serial_number} ")
                except Exception as e:
                    print(f"[Error the camera] {serial_number}: {e}")

    def run(self, stop_signal: mp.Event):
        print("🎥 [Camera Runner] Lockless 버퍼 기반 스레드/프로세스 시작")
        self.buffer = CameraShm(shm_name=self.shm_name, is_owner=False)

        ctx = Context()
        ctx.set_device_changed_callback(self._on_device_changed)

        device_list = ctx.query_devices()
        if device_list.get_count() > 0:
            for i in range(device_list.get_count()):
                device = device_list.get_device_by_index(i)
                info = device.get_device_info()
                serial_number = info.get_serial_number()

                try:
                    cam = OrbbecCamera(
                        device=device,
                        color_shape=(1280, 720, 3),
                        depth_shape=(1280, 720, 1),
                        settings_path=self.settings_path,
                    )
                    cam.start()
                    self.cameras[serial_number] = cam
                    print(f"[Success connection] {serial_number} ")
                except Exception as e:
                    print(f"[Error the camera] {serial_number}: {e}")

        frame_count = 0
        fps_start_time = time.time()
        accumulated_latency_ms = 0.0

        try:
            while not stop_signal.is_set():
                if len(self.cameras) == 0:
                    time.sleep(0.01)
                    continue

                for serial_number, cam in list(self.cameras.items()):
                    if not cam.pipeline:
                        continue
                    loop_start_ns = time.time_ns()

                    try:
                        frames = cam.pipeline.wait_for_frames(100)
                        if frames is None:
                            continue

                        frames = self.align_filter.process(frames)
                        if not frames:
                            continue

                        color_frame = frames.get_color_frame()
                        depth_frame = frames.get_depth_frame()
                        if not color_frame or not depth_frame:
                            continue

                        ts = depth_frame.get_global_timestamp_us()

                        self.buffer.write(ts, color_frame.get_data(), depth_frame.get_data())
                        self.buffer.set_status(True)

                        loop_end_ns = time.time_ns()
                        current_latency_ms = (loop_end_ns - loop_start_ns) / 1_000_000.0
                        accumulated_latency_ms += current_latency_ms
                        frame_count += 1

                    except Exception as e:
                        print(f"⚠️ 프레임 수신 중 장치 이탈 감지 ({serial_number}): {e}")
                        broken_cam = self.cameras.pop(serial_number, None)
                        if broken_cam:
                            broken_cam.stop()
                            del broken_cam

                now_sec = time.time()
                if now_sec - fps_start_time >= 1.0 and frame_count > 0:
                    avg_latency = accumulated_latency_ms / frame_count
                    self.buffer.set_latency(avg_latency)

                    frame_count = 0
                    accumulated_latency_ms = 0.0
                    fps_start_time = now_sec

        except KeyboardInterrupt:
            pass

        finally:
            print("🧹 [Camera Runner] 자원 해제 시작...")
            for cam in list(self.cameras.values()):
                cam.stop()
            self.cameras.clear()
            del ctx

            if self.buffer:
                self.buffer.close()
            print("🎥 [Camera Runner] 카메라 프로세스 정상 종료")


def camera_runner(shm_name, stop_signal):
    manager = CameraManager(shm_name)
    manager.run(stop_signal)


if __name__ == "__main__":
    import multiprocessing as mp  # mp 정의 누락 대응
    import cv2

    mp.set_start_method("spawn", force=True)
    stop_signal = mp.Event()

    shm_name = "orbbec_frame_buffer"
    camera_buffer = CameraShm(shm_name=shm_name, is_owner=True)

    camera_process = mp.Process(target=camera_runner, args=(shm_name, stop_signal))
    camera_process.start()

    prev_ts = None
    try:
        print("🚀 [Main Controller] 카메라 프로세스 실행 중 (종료하려면 'q' 또는 Ctrl+C)")
        while True:
            if camera_buffer.get_status():
                frame = camera_buffer.read()
                if prev_ts == frame.timestamp:
                    continue
                prev_ts = frame.timestamp

                # 1. Color 프레임 가져오기 (이미 BGR 포맷이라고 가정)
                color_img = frame.color

                # 2. Depth 프레임 전처리 (16비트 -> 8비트 시각화용 변환)
                depth_img = frame.depth
                # 0~5000mm(5m) 사이의 거리를 0~255 값으로 정규화 (카메라 스펙에 맞게 조절 가능)
                depth_clipped = np.clip(depth_img, 0, 5000)
                depth_normalized = cv2.normalize(
                    depth_clipped,
                    None,
                    0,
                    255,
                    cv2.NORM_MINMAX,
                    dtype=cv2.CV_8U,
                )
                # 깊이감을 보기 좋게 JET 컬러맵 적용 (가까운 곳은 빨간색/파란색 등)
                depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

                # 3. OpenCV 윈도우 표시

                color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
                cv2.imshow("Color Stream", color_img)
                cv2.imshow("Depth Stream", depth_colored)
                print(camera_buffer.get_latency())
                # 키 입력 처리 ('q' 누르면 안전 종료)
                if cv2.waitKey(1) & 0xFF == ord("q"):
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
