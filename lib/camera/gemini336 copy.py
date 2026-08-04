import time
import numpy as np
import multiprocessing as mp
from pathlib import Path
from typing import Optional, Tuple, Dict

from pyorbbecsdk import (
    Pipeline, Config, Context, AlignFilter,
    OBSensorType, OBFormat, OBPropertyID, OBStreamType
)

# --- Configuration Constants ---
CURRENT_DIR = Path(__file__).resolve().parent
PRESET_JSON_PATH = str((CURRENT_DIR / "gemini336_settings.json").resolve())

CHECK_PARAMS = [
    ("color_auto_exposure", OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, "bool"),
    ("color_auto_white_balance", OBPropertyID.OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL, "bool"),
    ("color_backlight_compensation", OBPropertyID.OB_PROP_COLOR_BACKLIGHT_COMPENSATION_INT, "int"),
    ("color_brightness", OBPropertyID.OB_PROP_COLOR_BRIGHTNESS_INT, "int"),
    ("color_contrast", OBPropertyID.OB_PROP_COLOR_CONTRAST_INT, "int"),
    ("color_exposure_time", OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT, "int"),
    ("color_gain", OBPropertyID.OB_PROP_COLOR_GAIN_INT, "int"),
    ("color_gamma", OBPropertyID.OB_PROP_COLOR_GAMMA_INT, "int"),
    ("color_hue", OBPropertyID.OB_PROP_COLOR_HUE_INT, "int"),
    ("color_power_line_frequency", OBPropertyID.OB_PROP_COLOR_POWER_LINE_FREQUENCY_INT, "int"),
    ("color_saturation", OBPropertyID.OB_PROP_COLOR_SATURATION_INT, "int"),
    ("color_sharpness", OBPropertyID.OB_PROP_COLOR_SHARPNESS_INT, "int"),
    ("color_white_balance", OBPropertyID.OB_PROP_COLOR_WHITE_BALANCE_INT, "int"),
    ("depth_auto_exposure", OBPropertyID.OB_PROP_DEPTH_AUTO_EXPOSURE_BOOL, "bool"),
    ("depth_exposure_time", OBPropertyID.OB_PROP_DEPTH_EXPOSURE_INT, "int"),
    ("depth_gain", OBPropertyID.OB_PROP_DEPTH_GAIN_INT, "int"),
    ("laser_state", OBPropertyID.OB_PROP_LASER_CONTROL_INT, "int"),
    ("laser_power_level", OBPropertyID.OB_PROP_LASER_POWER_LEVEL_CONTROL_INT, "int"),
    ("HW_align", OBPropertyID.OB_PROP_DEPTH_ALIGN_HARDWARE_BOOL, "bool"),
]

class OrbbecCamera:
    def __init__(self, device):
        self.device = device
        self.info = device.get_device_info()
        self.serial_number = self.info.get_serial_number()
        self.pipeline: Optional[Pipeline] = None

    def check_parameters(self):
        """기존 check_camera_parameters 역할"""
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
        """기존 print_camera_intrinsics 역할"""
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
            print(f"  ▶ Distortion Model : {c_dist.k1} {c_dist.k2} {c_dist.k3} {c_dist.k4} {c_dist.k5} {c_dist.k6} {c_dist.p1} {c_dist.p2}")

            d_intrin = param.depth_intrinsic
            print("-" * 50)
            print(" [Depth Camera Intrinsic Parameters]")
            print(f"  ▶ Resolution : {d_intrin.width} x {d_intrin.height}")
            print(f"  ▶ Focal Length (fx, fy) : ({d_intrin.fx:.4f}, {d_intrin.fy:.4f})")
            print(f"  ▶ Principal Point (cx, cy) : ({d_intrin.cx:.4f}, {d_intrin.cy:.4f})")
            d_dist = param.depth_distortion
            print(f"  ▶ Distortion Model : {d_dist.k1} {d_dist.k2} {d_dist.k3} {d_dist.k4} {d_dist.k5} {d_dist.k6} {d_dist.p1} {d_dist.p2}")
            print("=" * 50 + "\n")

        except Exception as e:
            print(f" 🛑 내적 파라미터(Intrinsic) 조회 실패: {e}")

    def start(self):
        json_path = Path(PRESET_JSON_PATH)
        print(f"🔍 [Preset Check] 경로: {json_path}")
        print(f"🔍 [Preset Check] 존재 여부: {json_path.exists()}, 크기: {json_path.stat().st_size if json_path.exists() else 0} bytes")

        """기존 setting_device 순서 정확히 준수"""
        # [순서 핵심 1] Pipeline 생성 및 Start 전에 반드시 Preset 먼저 로드!
        try:
            self.device.load_preset_from_json_file(PRESET_JSON_PATH)
            print(f"✅ 프리셋 로드 성공: {json_path.name}")
        except Exception as e:
            print(f"프리셋 로드 실패 ({self.serial_number}): {e}")

        # [순서 핵심 2] Pipeline 생성 및 스트림 프로필 설정
        self.pipeline = Pipeline(self.device)
        color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = color_profiles.get_video_stream_profile(COLOR_WIDTH, COLOR_HEIGHT, OBFormat.RGB, COLOR_FPS)

        depth_profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = depth_profiles.get_video_stream_profile(DEPTH_WIDTH, DEPTH_HEIGHT, OBFormat.Y16, DEPTH_FPS)

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
    """카메라 멀티 관리 및 프로세스 실행기"""
    def __init__(self, shm_name: str):
        self.shm_name = shm_name
        self.cameras: Dict[str, OrbbecCamera] = {}
        self.buffer: Optional[LocklessBuffer] = None
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

        if added_list.get_count() > 0:
            print("Added the camera:")
            for i in range(added_list.get_count()):
                device = added_list.get_device_by_index(i)
                info = device.get_device_info()
                serial_number = info.get_serial_number()
                device_name = info.get_name()

                print(f"[{i}] {device_name} (SN: {serial_number})")
                try:
                    cam = OrbbecCamera(device)
                    cam.start()
                    self.cameras[serial_number] = cam
                    print(f"[Success connection] {serial_number} ")
                except Exception as e:
                    print(f"[Error the camera] {serial_number}: {e}")

    def run(self, stop_signal: mp.Event):
        print("🎥 [Camera Runner] Lockless 버퍼 기반 스레드/프로세스 시작")
        self.buffer = LocklessBuffer(name=self.shm_name, is_owner=False)

        ctx = Context()
        ctx.set_device_changed_callback(self._on_device_changed)

        device_list = ctx.query_devices()
        if device_list.get_count() > 0:
            for i in range(device_list.get_count()):
                device = device_list.get_device_by_index(i)
                info = device.get_device_info()
                serial_number = info.get_serial_number()

                try:
                    cam = OrbbecCamera(device)
                    cam.start()
                    self.cameras[serial_number] = cam
                    print(f"[Success connection] {serial_number} ")
                except Exception as e:
                    print(f"[Error the camera] {serial_number}: {e}")

        try:
            while not stop_signal.is_set():
                if len(self.cameras) == 0:
                    time.sleep(0.01)
                    continue

                for serial_number, cam in list(self.cameras.items()):
                    if not cam.pipeline:
                        continue
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

                        ts = float(color_frame.get_timestamp_us()) / 1_000_000.0

                        # Lock 없이 즉시 쓰기
                        self.buffer.write(ts, color_frame.get_data(), depth_frame.get_data())

                    except Exception as e:
                        print(f"⚠️ 프레임 수신 중 장치 이탈 감지 ({serial_number}): {e}")
                        broken_cam = self.cameras.pop(serial_number, None)
                        if broken_cam:
                            broken_cam.stop()
                            del broken_cam

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


def runner(shm_name, stop_signal):
    manager = CameraManager(shm_name)
    manager.run(stop_signal)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    # 루트 컨트롤러에서 Shared Memory(Owner)와 Stop Event 제어
    main_buffer = LocklessBuffer(name=SHM_NAME, is_owner=True)
    stop_signal = mp.Event()

    cam_process = mp.Process(target=runner, args=(SHM_NAME, stop_signal))
    cam_process.start()

    try:
        print("🚀 [Main Controller] 카메라 프로세스 실행 중 (종료하려면 Ctrl+C)")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n종료 신호 수신. 카메라 프로세스를 정리합니다...")
        stop_signal.set()
        cam_process.join(timeout=3)
        if cam_process.is_alive():
            cam_process.terminate()

        main_buffer.close()
        print("모든 자원이 정상 해제되었습니다.")