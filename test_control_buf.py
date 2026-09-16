from multiprocessing import shared_memory
import numpy as np

SHM_NAME = "control_buf"
SIZE_IN_BYTES = 7 * np.dtype(np.float64).itemsize
control_shm = shared_memory.SharedMemory(create=False, size=SIZE_IN_BYTES, name=SHM_NAME)
control_buf = np.ndarray((7,), np.float64, buffer=control_shm.buf)


while True:
    d = 0.00001
    control_buf[0] -= d
    control_buf[6] = 1
    print(control_buf)
