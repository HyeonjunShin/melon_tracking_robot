import numpy as np
from multiprocessing import shared_memory
import ctypes
from dataclasses import dataclass


class BufferHeader(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_bool),
        ("index", ctypes.c_int64),
        ("latency", ctypes.c_double),
    ]


@dataclass
class PoseFrame:
    timestamp: int
    tf: np.ndarray


class DetectionBuffer:
    def __init__(
        self,
        shm_name: str,
        is_owner: bool = False,
    ):
        self.shm_name = shm_name
        self.is_owner = is_owner

        self.header_bytes = ctypes.sizeof(BufferHeader)
        self.timestamp_bytes = int(np.uint64().itemsize)
        self.tf_byte = int(4 * 4 * np.float64().itemsize)

        self.frame_bytes = self.timestamp_bytes + self.tf_byte
        self.total_bytes = self.header_bytes + (self.frame_bytes * 2)

        if self.is_owner:
            try:
                old_shm = shared_memory.SharedMemory(name=self.shm_name)
                old_shm.close()
                old_shm.unlink()
            except FileNotFoundError:
                pass
            self.shm = shared_memory.SharedMemory(name=self.shm_name, create=True, size=self.total_bytes)
        else:
            self.shm = shared_memory.SharedMemory(name=self.shm_name, create=False)

        self.header = BufferHeader.from_buffer(self.shm.buf)

        self.slots = []
        for i in range(2):
            slot_offset = self.header_bytes + (i * self.frame_bytes)
            tf_offset = slot_offset + self.timestamp_bytes

            ts_arr = np.ndarray(
                (1,),
                dtype=np.uint64,
                buffer=self.shm.buf,
                offset=slot_offset,
            )

            tf_arr = np.ndarray(
                (4, 4),
                dtype=np.float64,
                buffer=self.shm.buf,
                offset=tf_offset,
            )

            self.slots.append({"ts": ts_arr, "tf": tf_arr})

    def write(
        self,
        timestamp: np.uint64,
        tf_data: np.ndarray,
    ):
        next_idx = 1 - self.header.index
        target_slot = self.slots[next_idx]

        target_slot["ts"][0] = timestamp

        if tf_data is not None:
            tf_data_f64 = tf_data.view(np.float64)
            np.copyto(target_slot["tf"], tf_data_f64.reshape(4, 4))

        self.header.index = next_idx

    def get_latest_frame(self) -> PoseFrame:
        latest_idx = self.header.index
        slot = self.slots[latest_idx]

        return PoseFrame(
            timestamp=int(slot["ts"][0]),
            tf=slot["tf"].copy(),
        )

    def set_status(self, is_good: bool):
        self.header.status = is_good

    def get_status(self) -> bool:
        return self.header.status

    def set_latency(self, latency):
        self.header.latency = latency

    def get_latency(self):
        return self.header.latency

    def close(self):
        import gc

        self.slots.clear()
        gc.collect()

        self.shm.close()
        if self.is_owner:
            try:
                self.shm.unlink()
            except FileNotFoundError:
                pass
