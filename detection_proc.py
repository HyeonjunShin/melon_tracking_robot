import numpy as np
import multiprocessing as mp
from scipy.spatial.transform import Rotation as R

from lib.camera.gemini336 import Gemini336
from lib.camera.buffer import CameraBuffer



T_FLANGE_CAMERA = np.array(
    [
        [0.0, -0.93969262, 0.34202014, 0.05991575],
        [1.0, 0.0, 0.0, 0.00358377],
        [-0.0, 0.34202014, 0.93969262, 0.03416166],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)









def main():
    import time
    import cv2

    mp.set_start_method("spawn", force=True)
    stop_signal = mp.Event()
    frame_ready_signal = mp.Event()

    camera_shm_name = "camera_buffer"
    camera_buffer = CameraBuffer(shm_name=camera_shm_name, is_owner=True)
    camera_process = mp.Process(target=camera_runner, args=(camera_shm_name, stop_signal, frame_ready_signal))
    camera_process.start()

    detector_shm_name = "detection_buffer"
    detection_buffer = DetectorBuffer(shm_name=detector_shm_name, is_owner=True)
    detector_process = mp.Process(
        target=detector_runner, args=(camera_shm_name, detector_shm_name, stop_signal, frame_ready_signal)
    )
    detector_process.start()

    try:
        while True:
            if camera_buffer.get_status():
                break
            time.sleep(1)
        while True:
            if detection_buffer.get_status():
                break
            time.sleep(1)

        while not stop_signal.is_set():
            pass

    except KeyboardInterrupt:
        print("\n종료 신호 수신. 카메라 프로세스를 정리합니다...")
        stop_signal.set()

    finally:
        print("자원 해제 및 프로세스 종료 중...")

        camera_process.join(timeout=3)
        if camera_process.is_alive():
            camera_process.terminate()

        detector_process.join(timeout=3)
        if detector_process.is_alive():
            detector_process.terminate()

        # control_process.join(timeout=3)
        # if control_process.is_alive():
        #     control_process.terminate()

        cv2.destroyAllWindows()
        camera_buffer.close()
        detection_buffer.close()
        print("모든 자원이 정상 해제되었습니다.")


if __name__ == "__main__":
    main()
