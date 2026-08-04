import numpy as np
from multiprocessing import shared_memory

# COLOR_WIDTH, COLOR_HEIGHT, COLOR_FPS = 1280, 720, 30
# DEPTH_WIDTH, DEPTH_HEIGHT, DEPTH_FPS = 1280, 720, 30
# SHM_NAME = "orbbec_frame_buffer"

# TIMESTAMP_BYTES = int(np.float64().itemsize)
# COLOR_BYTES = int(np.prod((COLOR_HEIGHT, COLOR_WIDTH, 3)) * np.uint8().itemsize)
# DEPTH_BYTES = int(np.prod((DEPTH_HEIGHT, DEPTH_WIDTH, 1)) * np.uint16().itemsize)
# FRAME_BYTES = TIMESTAMP_BYTES + COLOR_BYTES + DEPTH_BYTES

class Frame:
    def __init__(self, timestamp: float, color: np.ndarray, depth: np.ndarray):
        self.timestamp = timestamp
        self.color = color
        self.depth = depth

    def __repr__(self):
        return f"<Frame TS:{self.timestamp:.2f} | Color:{self.color.shape} | Depth:{self.depth.shape}>"

class LocklessBuffer:
    def __init__(self, 
                 shm_name : str, 
                 is_owner : bool = False, 
                 color_shape : tuple = (720, 1280, 3), 
                 depth_shape : tuple = (720, 1280, 1)):
        
        self.shm_name        = shm_name
        self.is_owner        = is_owner

        self.color_shape     = color_shape
        self.depth_shape     = depth_shape

        self.header_bytes    = 16
        self.timestamp_bytes = int(np.uint64().itemsize)
        self.color_bytes     = int(np.prod(color_shape) * np.uint8().itemsize)
        self.depth_bytes     = int(np.prod(depth_shape) * np.uint16().itemsize)
        self.frame_bytes     = self.timestamp_bytes + self.color_bytes + self.depth_bytes
        self.total_bytes     = self.header_bytes + (self.frame_bytes * 2)

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

        self.status_arr      = np.ndarray((1,), dtype=np.bool_, buffer=self.shm.buf, offset=0)
        self.write_index_arr = np.ndarray((1,), dtype=np.int64, buffer=self.shm.buf, offset=8)

        self.slots = []
        for i in range(2):
            slot_offset = self.header_bytes + (i * self.frame_bytes)
            # ts_offset = slot_offset
            color_offset = slot_offset + self.timestamp_bytes
            depth_offset = slot_offset + self.timestamp_bytes + self.color_bytes

            ts_arr    = np.ndarray((1,),             dtype=np.uint64,  buffer=self.shm.buf, offset=slot_offset)
            color_arr = np.ndarray(self.color_shape, dtype=np.uint8,   buffer=self.shm.buf, offset=color_offset)
            depth_arr = np.ndarray(self.depth_shape, dtype=np.uint16,  buffer=self.shm.buf, offset=depth_offset)
            self.slots.append({"ts": ts_arr, "color": color_arr, "depth": depth_arr})

    def write(self, timestamp: np.uint64, color_data: np.ndarray, depth_data: np.ndarray):
        current_idx = int(self.write_index_arr[0])
        next_idx = 1 - current_idx

        target_slot = self.slots[next_idx]
        target_slot["ts"][0] = timestamp

        if color_data is not None:
            np.copyto(target_slot["color"], color_data.reshape(self.color_shape))

        if depth_data is not None:
            depth_data_u16 = depth_data.view(np.uint16)
            np.copyto(target_slot["depth"], depth_data_u16.reshape(self.depth_shape))

        self.write_index_arr[0] = next_idx

    def read_latest_frame(self) -> Frame:
        latest_idx = int(self.write_index_arr[0])
        slot = self.slots[latest_idx]

        return Frame(
            timestamp = int(slot["ts"][0]),
            color     = slot["color"].copy(),
            depth     = slot["depth"].copy(),
        )

    def set_status(self, is_good:bool):
        self.status_arr[0] = is_good

    def get_status(self) -> bool:
        return bool(self.status_arr[0])

    def close(self):
        self.shm.close()
        if self.is_owner:
            try:
                self.shm.unlink()
            except FileNotFoundError:
                pass

