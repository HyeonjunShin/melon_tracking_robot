import mmap
import struct
import numpy as np
from scipy.spatial.transform import Rotation as R


class FlangeBuffer:
    def __init__(self, shm_name="/flange_pose"):
        self.shm_path = f"/dev/shm{shm_name}"
        self.file = open(self.shm_path, "r+b")
        self.shm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)

        self.buffer_size = 100
        self.header_size = 16
        frame_dtype = np.dtype([("timestamp_us", np.int64), ("flange_pos", np.float64, (6,))])

        # Zero-copy 매핑
        self.buffer = np.frombuffer(
            self.shm, dtype=frame_dtype, count=self.buffer_size, offset=self.header_size
        )

    def get_ordered_data(self):
        """가장 오래된 데이터(index 0) -> 최신 데이터(index -1) 순 정렬"""
        sequence, head = struct.unpack("QI", self.shm[:12])
        if sequence == 0:
            return None, 0

        if sequence < self.buffer_size:
            ordered_data = self.buffer[:sequence]
        else:
            # 링 버퍼 재정렬
            ordered_data = np.roll(self.buffer, -head, axis=0)
        return ordered_data, sequence

    def get_flange_tf(self, target_ts_us: int):
        sequence, head = struct.unpack("QI", self.shm[:12])

        if sequence == 0:
            return {"target_ts": target_ts_us, "matched_ts": 0, "time_diff_us": 0, "T_base_flange": np.eye(4)}

        valid_size = sequence if sequence < self.buffer_size else self.buffer_size
        start_idx = 0 if sequence < self.buffer_size else head

        oldest_idx = start_idx
        newest_idx = (start_idx + valid_size - 1) % self.buffer_size

        oldest_ts = self.buffer[oldest_idx]["timestamp_us"]
        newest_ts = self.buffer[newest_idx]["timestamp_us"]

        # 1. target_ts가 버퍼 범위를 벗어난 경우 처리
        if target_ts_us <= oldest_ts:
            best_idx = oldest_idx
        elif target_ts_us >= newest_ts:
            best_idx = newest_idx
        else:
            # 2. 이진 탐색으로 target_ts 이상인 첫번째 위치(high) 탐색
            low = 0
            high = valid_size - 1

            while low < high:
                mid = (low + high) // 2
                real_mid_idx = (start_idx + mid) % self.buffer_size

                if self.buffer[real_mid_idx]["timestamp_us"] < target_ts_us:
                    low = mid + 1
                else:
                    high = mid

            # target_ts 직후(right)와 직전(left) 중 절댓값이 더 가까운 것 선택
            right_idx = (start_idx + high) % self.buffer_size
            left_idx = (start_idx + high - 1 + self.buffer_size) % self.buffer_size

            diff_right = abs(self.buffer[right_idx]["timestamp_us"] - target_ts_us)
            diff_left = abs(self.buffer[left_idx]["timestamp_us"] - target_ts_us)

            best_idx = right_idx if diff_right < diff_left else left_idx

        matched_frame = self.buffer[best_idx]
        matched_ts = matched_frame["timestamp_us"]
        pos = matched_frame["flange_pos"]

        # 3. 4x4 Transformation Matrix 생성 (보간 없음)
        T_base_flange = np.eye(4)
        # Position (XYZ)
        T_base_flange[:3, 3] = pos[:3]
        # Rotation (Rx, Ry, Rz - 두산 ZYZ Euler Angle)
        rot = R.from_euler("ZYZ", pos[3:], degrees=True)
        T_base_flange[:3, :3] = rot.as_matrix()

        return {
            "target_ts": target_ts_us,
            "matched_ts": int(matched_ts),
            "time_diff_us": int(abs(matched_ts - target_ts_us)),  # 실제 차이 난 시간 (us)
            "T_base_flange": T_base_flange,
        }

    def close(self):
        self.buffer = None
        self.shm.close()
        self.file.close()


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
    current_us = int(time.time() * 1e6)
    print(reader.get_flange_tf(current_us)["time_diff_us"] * 0.001)


if __name__ == "__main__":
    print_terminal_dashboard()
    # print_target_ts()
