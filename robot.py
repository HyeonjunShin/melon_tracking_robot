from lib.control.ik_py import PyIk
from lib.control.dsr_py import DoosanRobotController
import threading
import time
import numpy as np
from multiprocessing import shared_memory

URDF = "/home/uon/Downloads/ik_solver-dev/data/robot/urdf/doosan_m1013.urdf"
solver = PyIk(URDF)
is_init_ik = False
robot = DoosanRobotController("192.168.1.30", 500)
# INIT_POSE = [0.0, 0.9, 0.5, 180, 180, -90 + 90]
# TARGET_POSE = INIT_POSE

SHM_NAME = "control_buf"
SIZE_IN_BYTES = 7 * np.dtype(np.float64).itemsize
control_shm = shared_memory.SharedMemory(create=True, size=SIZE_IN_BYTES, name=SHM_NAME)
control_buf = np.ndarray((7,), np.float64, buffer=control_shm.buf)
INIT_POSE = [0.0, 0.9, 0.5, 180, 180, -90 + 90]
control_buf[:6] = INIT_POSE
TARGET_POSE = control_buf[:6]
STATE = control_buf[6:]


TOOL_CAM = np.array(
    [
        [0.0, -0.93969, 0.34202, 0.06146],
        [1.0, 0.0, 0.0, 0.00100],
        [0.0, 0.34202, 0.93969, 0.03085],
        [0.0, 0.0, 0.0, 1.00000],
    ],
    dtype=np.float64,
)

TOOL_SCUTION = np.array(
    [
        [0.0, -1.0, 0.0, 0.000],
        [1.0, 0.0, 0.0, 0.000],
        [0.0, 0.0, 1.0, 0.255],
        [0.0, 0.0, 0.0, 1.000],
    ],
    dtype=np.float64,
)


def ik_callback():
    global is_init_ik, robot, solver, TARGET_POSE

    while True:
        if not is_init_ik:
            continue

        target_pose = TARGET_POSE.copy()  # TCP Pose

        target_matrix = PyIk.make_tf(
            target_pose[0],
            target_pose[1],
            target_pose[2],  # x, y, z [m]
            target_pose[3],
            target_pose[4],
            target_pose[5],  # roll, pitch, yaw [rad]
            use_deg=True,
        )

        solver.movel(target_matrix)  # This Must be in threding
        time.sleep(0.001)


def main():
    global is_init_ik, robot, solver, TARGET_POSE, STATE

    robot.connect()
    time.sleep(0.1)
    robot.servo_on()
    time.sleep(3.0)
    robot.start_rt()

    if not solver.init():
        print("Error: PyIk 초기화 실패")
        return
    time.sleep(1.0)

    init_joint = np.zeros((7,))
    init_joint[:6] = robot.get_curr_joint_deg()
    solver.set_joint(init_joint, use_deg=True)

    solver.set_end_effector_offset(TOOL_CAM)
    # solver.set_tcp_max_speed(0.1)

    is_init_ik = True
    th = threading.Thread(target=ik_callback)
    th.start()

    while True:
        if STATE == 0:
            control_buf[:6] = INIT_POSE
            solver.set_end_effector_offset(TOOL_CAM)

        if STATE == 1:
            solver.set_end_effector_offset(TOOL_SCUTION)

        res = solver.get_current_joint(use_deg=True)
        robot.movej_rt(res[0:6], 0.001)
        time.sleep(0.001)


if __name__ == "__main__":
    main()
