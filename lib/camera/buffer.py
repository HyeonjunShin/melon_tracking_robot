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
class Frame:
    timestamp: int
    color: np.ndarray
    depth: np.ndarray


class CameraBuffer:
    def __init__(
        self,
        shm_name: str,
        is_owner: bool = False,
        color_shape: tuple = (720, 1280, 3),
        depth_shape: tuple = (720, 1280, 1),
    ):

        self.shm_name = shm_name
        self.is_owner = is_owner

        self.color_shape = color_shape
        self.depth_shape = depth_shape

        self.header_bytes = ctypes.sizeof(BufferHeader)
        self.timestamp_bytes = int(np.uint64().itemsize)
        self.color_bytes = int(np.prod(color_shape) * np.uint8().itemsize)
        self.depth_bytes = int(np.prod(depth_shape) * np.uint16().itemsize)
        self.frame_bytes = self.timestamp_bytes + self.color_bytes + self.depth_bytes
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
            # ts_offset = slot_offset
            color_offset = slot_offset + self.timestamp_bytes
            depth_offset = slot_offset + self.timestamp_bytes + self.color_bytes

            ts_arr = np.ndarray(
                (1,),
                dtype=np.uint64,
                buffer=self.shm.buf,
                offset=slot_offset,
            )
            color_arr = np.ndarray(
                self.color_shape,
                dtype=np.uint8,
                buffer=self.shm.buf,
                offset=color_offset,
            )
            depth_arr = np.ndarray(
                self.depth_shape,
                dtype=np.uint16,
                buffer=self.shm.buf,
                offset=depth_offset,
            )
            self.slots.append({"ts": ts_arr, "color": color_arr, "depth": depth_arr})

    def write(
        self,
        timestamp: np.uint64,
        color_data: np.ndarray,
        depth_data: np.ndarray,
    ):
        next_idx = 1 - self.header.index
        target_slot = self.slots[next_idx]

        target_slot["ts"][0] = timestamp

        if color_data is not None:
            np.copyto(target_slot["color"], color_data.reshape(self.color_shape))

        if depth_data is not None:
            depth_data_u16 = depth_data.view(np.uint16)
            np.copyto(target_slot["depth"], depth_data_u16.reshape(self.depth_shape))

        self.header.index = next_idx

    def read_latest_frame(self) -> Frame:
        latest_idx = self.header.index
        slot = self.slots[latest_idx]

        return Frame(
            timestamp=int(slot["ts"][0]),
            color=slot["color"],
            depth=slot["depth"],
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
        del self.header
        self.slots.clear()

        import gc

        gc.collect()

        self.shm.close()
        if self.is_owner:
            try:
                self.shm.unlink()
            except FileNotFoundError:
                pass
