import time
import threading
import math
import numpy as np  # 💡 행렬 조작을 위해 넘파이 임포트

from ik_py import PyIk
from dsr_py import DoosanRobotController

np.set_printoptions(suppress=True, precision=4)

# ----------------------------------------------
# IK solver 생성
# ----------------------------------------------
URDF = "/home/uon/Downloads/ik_solver-dev/data/robot/urdf/doosan_m1013.urdf"
solver = PyIk(URDF)
is_init_ik = False

INIT_JOINT = [-130.50, -6.62, -88.98, 0.08, -84.39, 66.5, 0]  # [deg]
# INIT_JOINT = np.zeros((7,))
TARGET_POSE = [0.43742, 0.45275, 0.5, 38.92, -180, -123.24]  # x, y, z, roll, pitch, yaw [m, deg]
# TARGET_POSE = [0.43742, 0.45275, 0.59375, 0, 0, 0]  # x, y, z, roll, pitch, yaw [m, deg]
# TARGET_POSE = [0, 0, 0, 0, 0, 0]  # x, y, z, roll, pitch, yaw [m, deg]

robot = DoosanRobotController("192.168.1.30", 500)


def ik_callback():
    global is_init_ik, robot, solver, INIT_JOINT, TARGET_POSE
    d = 0.0005  # 1mm씩 움직임

    while True:
        if not is_init_ik:
            continue

        if TARGET_POSE[2] < 0.3 or TARGET_POSE[2] > 0.7:
            d *= -1
        TARGET_POSE[2] = TARGET_POSE[2] + d

        target_pose = TARGET_POSE  # TCP Pose

        target_matrix = PyIk.make_tf(
            target_pose[0],
            target_pose[1],
            target_pose[2],  # x, y, z [m]
            target_pose[3],
            target_pose[4],
            target_pose[5],  # roll, pitch, yaw [rad]
            use_deg=False,
        )

        solver.movel(target_matrix)  # This Must be in threding
        time.sleep(0.001)


def main():
    global is_init_ik, robot, solver, INIT_JOINT, TARGET_POSE

    # target_pose = TARGET_POSE
    # target_matrix = PyIk.make_tf(
    #     target_pose[0],
    #     target_pose[1],
    #     target_pose[2],  # x, y, z [m]
    #     target_pose[3],
    #     target_pose[4],
    #     target_pose[5],  # roll, pitch, yaw [rad]
    #     use_deg=False,
    # )

    # while 1:
    # solver.movel(target_matrix)
    # print(solver.get_current_joint())
    # print(target_matrix)

    # # Robot On
    robot.connect()
    time.sleep(0.1)
    robot.servo_on()
    time.sleep(3.0)
    robot.start_rt()

    if not solver.init():
        print("Error: PyIk 초기화 실패")
        return
    time.sleep(1.0)
    INIT_JOINT[:6] = robot.get_curr_joint_deg()
    # robot.movej(INIT_JOINT, 3.0)  # 3 sec moving
    solver.set_joint(INIT_JOINT, use_deg=True)
    is_init_ik = True

    # # RT Mode Start
    # time.sleep(3.0)

    # for i in range(100):
    #     target_pose = TARGET_POSE
    #     target_matrix = PyIk.make_tf(
    #         target_pose[0],
    #         target_pose[1],
    #         target_pose[2],  # x, y, z [m]
    #         target_pose[3],
    #         target_pose[4],
    #         target_pose[5],  # roll, pitch, yaw [rad]
    #         use_deg=False,
    #     )

    #     solver.movel(target_matrix)
    #     print(solver.get_current_joint(use_deg=True))

    # ----------------------------------------------
    # 초기 조인트 값 설정
    # 기본적으로 7개의 조인트값을 입력함 7축 로봇 URDF가 아니라면 마지막 값은 쓰레기값으로 처리됨
    # ----------------------------------------------

    # # solver.set_workspace_limits((-1, 1)(-1, 1), (-1, 1))
    # # 3번째 조인트의 동작 범위 설정

    th = threading.Thread(target=ik_callback)
    th.start()

    # Robot Control Loop
    while True:
        # curr_tf = robot.get_flange_tf(time.time_ns())
        # print(curr_tf)

        res = solver.get_current_joint(use_deg=True)

        robot.movej_rt(res[0:6], 0.001)
        time.sleep(0.001)


if __name__ == "__main__":
    main()
