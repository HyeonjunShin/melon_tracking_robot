import time
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
from pyorbbecsdk import Pipeline, Config, Context
from pyorbbecsdk import (
    OBSensorType,
    OBFormat,
    OBPropertyID,
    AlignFilter,
    OBStreamType,
)

PRESET_JSON_PATH = "/home/uon/temp/gg_version/gemini336/gemini336_settings.json"
COLOR_WIDTH = 1280
COLOR_HEIGHT = 720
COLOR_FPS = 30
DEPTH_WIDTH = 1280
DEPTH_HEIGHT = 720
DEPTH_FPS = 30
SHM_NAME = "orbbec_frame_buffer"

# 슬롯 1개당 프레임 크기 (Timestamp + Color + Depth)
TIMESTAMP_BYTES = int(np.float64().itemsize)
COLOR_BYTES = int(np.prod((COLOR_HEIGHT, COLOR_WIDTH, 3)) * np.uint8().itemsize)
DEPTH_BYTES = int(np.prod((DEPTH_HEIGHT, DEPTH_WIDTH, 1)) * np.uint16().itemsize)
FRAME_BYTES = TIMESTAMP_BYTES + COLOR_BYTES + DEPTH_BYTES

CHECK_PARAMS = [
    ("color_auto_exposure", OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, "bool"),
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
    ("color_white_balance", OBPropertyID.OB_PROP_COLOR_WHITE_BALANCE_INT, "int"),
    ("depth_auto_exposure", OBPropertyID.OB_PROP_DEPTH_AUTO_EXPOSURE_BOOL, "bool"),
    ("depth_exposure_time", OBPropertyID.OB_PROP_DEPTH_EXPOSURE_INT, "int"),
    ("depth_gain", OBPropertyID.OB_PROP_DEPTH_GAIN_INT, "int"),
    ("laser_state", OBPropertyID.OB_PROP_LASER_CONTROL_INT, "int"),
    ("laser_power_level", OBPropertyID.OB_PROP_LASER_POWER_LEVEL_CONTROL_INT, "int"),
    ("HW_align", OBPropertyID.OB_PROP_DEPTH_ALIGN_HARDWARE_BOOL, "bool"),
]

PIPELINES = {}


class Frame:
    def __init__(self, timestamp: float, color: np.ndarray, depth: np.ndarray):
        self.timestamp = timestamp
        self.color = color
        self.depth = depth

    def __repr__(self):
        return f"<Frame TS:{self.timestamp:.2f} | Color:{self.color.shape} | Depth:{self.depth.shape}>"


class LocklessBuffer:
    """
    Lock 없이 Write Index(0 또는 1)로 2개의 슬롯을 교체하며
    카메라 수신부의 대기 시간(Latency)을 0ms로 만들어주는 Double Buffer
    """

    def __init__(self, name: str = SHM_NAME, is_owner: bool = False):
        self.shm_name = name
        self.is_owner = is_owner

        # Header(8 byte: Write Index) + Slot 0 (Frame) + Slot 1 (Frame)
        self.header_bytes = 8
        self.total_bytes = self.header_bytes + (FRAME_BYTES * 2)

        if self.is_owner:
            try:
                old_shm = shared_memory.SharedMemory(name=self.shm_name)
                old_shm.close()
                old_shm.unlink()
            except FileNotFoundError:
                pass
            self.shm = shared_memory.SharedMemory(
                name=self.shm_name, create=True, size=self.total_bytes
            )
        else:
            self.shm = shared_memory.SharedMemory(name=self.shm_name, create=False)

        # 1. Header: 생산자가 쓰기를 완료한 최신 슬롯 번호 (0 또는 1)
        self.write_index_arr = np.ndarray(
            (1,), dtype=np.int64, buffer=self.shm.buf, offset=0
        )

        # 2. Slot 0과 Slot 1의 memoryview 구조 분할
        self.slots = []
        for i in range(2):
            slot_offset = self.header_bytes + (i * FRAME_BYTES)

            ts_offset = slot_offset
            color_offset = slot_offset + TIMESTAMP_BYTES
            depth_offset = slot_offset + TIMESTAMP_BYTES + COLOR_BYTES

            ts_arr = np.ndarray(
                (1,), dtype=np.float64, buffer=self.shm.buf, offset=ts_offset
            )
            color_arr = np.ndarray(
                (COLOR_HEIGHT, COLOR_WIDTH, 3),
                dtype=np.uint8,
                buffer=self.shm.buf,
                offset=color_offset,
            )
            depth_arr = np.ndarray(
                (DEPTH_HEIGHT, DEPTH_WIDTH, 1),
                dtype=np.uint16,
                buffer=self.shm.buf,
                offset=depth_offset,
            )
            self.slots.append({"ts": ts_arr, "color": color_arr, "depth": depth_arr})

    def write(self, timestamp: float, color_data: np.ndarray, depth_data: np.ndarray):
        # 소비자가 읽는 슬롯의 반대편 슬롯에 락 없이 쓰기
        current_idx = int(self.write_index_arr[0])
        next_idx = 1 - current_idx  # 0 -> 1, 1 -> 0

        target_slot = self.slots[next_idx]
        target_slot["ts"][0] = timestamp

        if color_data is not None:
            color_reshaped = color_data.reshape((COLOR_HEIGHT, COLOR_WIDTH, 3))
            np.copyto(target_slot["color"], color_reshaped)

        if depth_data is not None:
            depth_data = depth_data.view(np.uint16)
            depth_reshaped = depth_data.reshape((DEPTH_HEIGHT, DEPTH_WIDTH, 1))
            np.copyto(target_slot["depth"], depth_reshaped)

        # 쓰기가 완전히 끝난 후 최신 슬롯 인덱스 업데이트 (Atomic)
        self.write_index_arr[0] = next_idx

    def read_latest_frame(self) -> Frame:
        # 생산자가 기록 완료한 최신 슬롯 인덱스를 가져와서 읽음
        latest_idx = int(self.write_index_arr[0])
        slot = self.slots[latest_idx]

        return Frame(
            timestamp=float(slot["ts"][0]),
            color=slot["color"].copy(),  # Zero-copy 뷰에서 독립 복사본 생성
            depth=slot["depth"].copy(),
        )

    def close(self):
        self.shm.close()
        if self.is_owner:
            try:
                self.shm.unlink()
            except FileNotFoundError:
                pass


def check_camera_parameters(device):
    def check_prop(label, prop_id, prop_type="int"):
        try:
            if prop_type == "bool":
                val = device.get_bool_property(prop_id)
            elif prop_type == "float":
                val = device.get_float_property(prop_id)
            else:
                val = device.get_int_property(prop_id)
            print(f" ▶ {label:<30} : {val}")
        except Exception:
            print(f" 🛑 {label:<30} : [조회 실패 / 지원하지 않는 프로퍼티]")

    for param in CHECK_PARAMS:
        check_prop(param[0], param[1], param[2])


def print_camera_intrinsics(pipeline):
    try:
        param = pipeline.get_camera_param()

        # Color Intrinsics
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

        # Depth Intrinsics
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


def setting_device(device, serial_number):
    try:
        device.load_preset_from_json_file(PRESET_JSON_PATH)
    except Exception as e:
        print(f"프리셋 로드 실패 ({serial_number}): {e}")

    pipeline = Pipeline(device)
    color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = color_profiles.get_video_stream_profile(
        COLOR_WIDTH, COLOR_HEIGHT, OBFormat.RGB, COLOR_FPS
    )
    depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = depth_profiles.get_video_stream_profile(
        DEPTH_WIDTH, DEPTH_HEIGHT, OBFormat.Y16, DEPTH_FPS
    )

    config = Config()
    config.enable_stream(color_profile)
    config.enable_stream(depth_profile)

    pipeline.start(config)
    PIPELINES[serial_number] = pipeline
    print_camera_intrinsics(pipeline)


def on_device_changed_callback(removed_list, added_list):
    if removed_list.get_count() > 0:
        print("Deleted the camera")
        for i in range(removed_list.get_count()):
            serial_number = removed_list.get_device_serial_number_by_index(i)
            pipeline = PIPELINES.pop(serial_number, None)
            if pipeline:
                try:
                    pipeline.stop()
                except Exception:
                    pass
                del pipeline
            print(
                f"  - {removed_list.get_device_name_by_index(i)} (SN: {serial_number})"
            )

    if added_list.get_count() > 0:
        print("Added the camera:")
        for i in range(added_list.get_count()):
            device = added_list.get_device_by_index(i)
            info = device.get_device_info()
            serial_number = info.get_serial_number()
            device_name = info.get_name()

            print(f"[{i}] {device_name} (SN: {serial_number})")
            try:
                setting_device(device, serial_number)
                check_camera_parameters(device)
                print(f"[Success connection] {serial_number} ")
            except Exception as e:
                print(f"[Error the camera] {serial_number}: {e}")


def runner(shm_name, stop_signal):
    print("🎥 [Camera Runner] Lockless 버퍼 기반 스레드/프로세스 시작")
    buffer = LocklessBuffer(name=shm_name, is_owner=False)

    ctx = Context()
    ctx.set_device_changed_callback(on_device_changed_callback)

    align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

    device_list = ctx.query_devices()
    if device_list.get_count() > 0:
        for i in range(device_list.get_count()):
            device = device_list.get_device_by_index(i)
            info = device.get_device_info()
            serial_number = info.get_serial_number()

            setting_device(device, serial_number)
            check_camera_parameters(device)

    while not stop_signal.is_set():
        if len(PIPELINES) == 0:
            time.sleep(0.01)
            continue

        for serial_number, p in list(PIPELINES.items()):
            try:
                frames = p.wait_for_frames(100)
                if frames is None:
                    continue

                frames = align_filter.process(frames)
                if not frames:
                    continue

                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                ts = float(color_frame.get_timestamp_us()) / 1_000_000.0

                # Lock 없이 0ms 즉시 기록
                buffer.write(ts, color_frame.get_data(), depth_frame.get_data())

            except Exception as e:
                print(f"⚠️ 프레임 수신 중 장치 이탈 감지 ({serial_number}): {e}")
                broken_p = PIPELINES.pop(serial_number, None)
                if broken_p:
                    try:
                        broken_p.stop()
                    except Exception:
                        pass
                    del broken_p

    for pipeline in list(PIPELINES.values()):
        try:
            pipeline.stop()
        except Exception:
            pass
    PIPELINES.clear()
    del ctx

    buffer.close()
    print("🎥 [Camera Runner] 카메라 프로세스 종료")


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
