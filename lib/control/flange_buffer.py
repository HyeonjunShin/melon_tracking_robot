import mmap
import struct
import numpy as np
from scipy.spatial.transform import Rotation as R
from multiprocessing import shared_memory


class FlangeBuffer:
    def __init__(self, shm_name="flange_pose", is_owner=False):
        self.buffer_size = 100
        self.header_size = 16
        self.data_frame_size = 8 + (8 * 6)  # 56 bytes
        self.shared_memory_size = 8 + 4 + 4 + (self.data_frame_size * self.buffer_size)
        self.shm_name = shm_name
        self.is_owner = is_owner

        if self.is_owner:
            try:
                old_shm = shared_memory.SharedMemory(name=self.shm_name)
                old_shm.close()
                old_shm.unlink()
            except FileNotFoundError:
                pass

            self.shm = shared_memory.SharedMemory(
                name=self.shm_name, create=True, size=self.shared_memory_size
            )
            self.shm.buf[: self.shared_memory_size] = b"\x00" * self.shared_memory_size  # init zero.
        else:
            self.shm = shared_memory.SharedMemory(
                name=self.shm_name, create=False, size=self.shared_memory_size
            )

        header_dtype = np.dtype([("sequence", np.uint64), ("head", np.uint32), ("padding", np.uint32)])
        self.header_buffer = np.frombuffer(self.shm.buf, dtype=header_dtype, count=1)[0]

        frame_dtype = np.dtype([("timestamp_us", np.int64), ("flange_pos", np.float64, (6,))])
        self.frame_buffer = np.frombuffer(
            self.shm.buf, dtype=frame_dtype, count=self.buffer_size, offset=self.header_size
        )

    def get_ordered_data(self):
        sequence = self.header_buffer["sequence"]
        head = self.header_buffer["head"]
        if sequence == 0:
            return None, 0

        if sequence < self.buffer_size:
            ordered_data = self.frame_buffer[:sequence]
        else:
            ordered_data = np.roll(self.frame_buffer, -head, axis=0)
        return ordered_data, sequence

    def get_flange_tf(self, target_ts_us: int):
        sequence = self.header_buffer["sequence"]
        head = self.header_buffer["head"]
        if sequence == 0:
            return {"target_ts": target_ts_us, "matched_ts": 0, "time_diff_us": 0, "T_base_flange": np.eye(4)}

        valid_size = sequence if sequence < self.buffer_size else self.buffer_size
        start_idx = 0 if sequence < self.buffer_size else head

        oldest_idx = start_idx
        newest_idx = (start_idx + valid_size - 1) % self.buffer_size

        oldest_ts = self.frame_buffer[oldest_idx]["timestamp_us"]
        newest_ts = self.frame_buffer[newest_idx]["timestamp_us"]

        if target_ts_us <= oldest_ts:
            best_idx = oldest_idx
        elif target_ts_us >= newest_ts:
            best_idx = newest_idx
        else:
            low = 0
            high = valid_size - 1

            while low < high:
                mid = (low + high) // 2
                real_mid_idx = (start_idx + mid) % self.buffer_size

                if self.frame_buffer[real_mid_idx]["timestamp_us"] < target_ts_us:
                    low = mid + 1
                else:
                    high = mid

            right_idx = (start_idx + high) % self.buffer_size
            left_idx = (start_idx + high - 1 + self.buffer_size) % self.buffer_size

            diff_right = abs(self.frame_buffer[right_idx]["timestamp_us"] - target_ts_us)
            diff_left = abs(self.frame_buffer[left_idx]["timestamp_us"] - target_ts_us)

            best_idx = right_idx if diff_right < diff_left else left_idx

        matched_frame = self.frame_buffer[best_idx]
        matched_ts = matched_frame["timestamp_us"]
        pos = matched_frame["flange_pos"]

        T_base_flange = np.eye(4)
        # Position (XYZ)
        T_base_flange[:3, 3] = pos[:3]
        # Rotation (두산 ZYZ Euler Angle)
        rot = R.from_euler("ZYZ", pos[3:], degrees=True)
        T_base_flange[:3, :3] = rot.as_matrix()

        return {
            "target_ts": target_ts_us,
            "matched_ts": int(matched_ts),
            "time_diff_us": int(abs(matched_ts - target_ts_us)),
            "T_base_flange": T_base_flange,
        }

    def close(self):
        self.header_buffer = None
        self.frame_buffer = None
        if hasattr(self, "shm") and self.shm is not None:
            self.shm.close()
            try:
                self.shm.unlink()
            except FileNotFoundError:
                pass
            self.shm = None


def print_terminal_dashboard():
    import time

    reader = FlangeBuffer()

    try:
        while True:
            data, seq = reader.get_ordered_data()

            if data is not None and len(data) > 0:
                latest = data[-1]
                oldest = data[0]
                dt_ms = (latest["timestamp_us"] - oldest["timestamp_us"]) / 1000.0

                # 터미널 커서를 맨 위로 이동 (화면 깜빡임 없는 실시간 갱신)
                print("\033[H\033[J", end="")

                print(
                    "=========================================================================================="
                )
                print(f" [DSR Robot SHM Monitor]  누적 프레임: {seq} | 가져온 버퍼 수: {len(data)}/500")
                print(f" 버퍼 시간 span: {dt_ms:.2f} ms | 최신 Timestamp: {latest['timestamp_us']} us")
                print(
                    "=========================================================================================="
                )
                print(
                    f" [최신 Pose]  X: {latest['flange_pos'][0]:8.2f} | Y: {latest['flange_pos'][1]:8.2f} | Z: {latest['flange_pos'][2]:8.2f} "
                    f"| Rx: {latest['flange_pos'][3]:7.2f} | Ry: {latest['flange_pos'][4]:7.2f} | Rz: {latest['flange_pos'][5]:7.2f}"
                )
                print(
                    "------------------------------------------------------------------------------------------"
                )

                # 데이터가 500개 이상일 때 처음 3개, 중간 생략, 최근 5개 샘플 출력
                print(" [최근 500개 데이터 샘플 목록 (시간순)]")
                print(
                    " Index | Timestamp (us) |   X (mm)   |   Y (mm)   |   Z (mm)   |   Rx (deg) |   Ry (deg) |   Rz (deg) "
                )
                print(
                    "-------+----------------+------------+------------+------------+------------+------------+------------"
                )

                show_indices = list(range(min(10, len(data))))
                if len(data) > 8:
                    show_indices.append(-1)  # 생략 표시용
                    show_indices.extend(range(len(data) - 10, len(data)))
                else:
                    show_indices = list(range(len(data)))

                has_printed_dots = False
                for idx in show_indices:
                    if idx == -1:
                        if not has_printed_dots:
                            print(
                                "  ...  |       ...      |    ...     |    ...     |    ...     |    ...     |    ...     |    ...    "
                            )
                            has_printed_dots = True
                        continue

                    row = data[idx]
                    p = row["flange_pos"]
                    print(
                        f" {idx:4d}  | {row['timestamp_us']:14d} | {p[0]:10.2f} | {p[1]:10.2f} | {p[2]:10.2f} | {p[3]:10.2f} | {p[4]:10.2f} | {p[5]:10.2f}"
                    )

                print(
                    "=========================================================================================="
                )
                print(" Ctrl+C를 누르면 종료됩니다.")

            time.sleep(0.01)  # 20Hz 터미널 출력 갱신

    except KeyboardInterrupt:
        print("\n모니터링을 종료합니다.")
    finally:
        reader.close()


def print_target_ts():
    import time

    reader = FlangeBuffer()
    while 1:
        current_us = int(time.time() * 1e6)
        print(reader.get_flange_tf(current_us)["time_diff_us"] * 0.001)
        print(reader.get_flange_tf(current_us)["T_base_flange"])


if __name__ == "__main__":
    SHM_NAME = "control_buf"
    try:
        old_shm = shared_memory.SharedMemory(name=SHM_NAME)
        old_shm.close()
        old_shm.unlink()
    except FileNotFoundError:
        pass
    SIZE_IN_BYTES = 7 * np.dtype(np.float64).itemsize
    control_shm = shared_memory.SharedMemory(create=True, size=SIZE_IN_BYTES, name=SHM_NAME)
    control_buf = np.ndarray((7,), np.float64, buffer=control_shm.buf)

    print_terminal_dashboard()
    # print_target_ts()
