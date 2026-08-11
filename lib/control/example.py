import time
import threading
import math

from ik_py import PyIk
from dsr_py import DoosanRobotController

# ----------------------------------------------
# IK solver 생성
# ----------------------------------------------
URDF = "/home/uon/Downloads/ik_solver-dev/data/robot/urdf/doosan_m1013.urdf"
solver = PyIk(URDF)
is_init_ik = False

INIT_JOINT = [-130, -8.46, -97.13, 0.01, -75.0, 49.46, 0]  # [deg]
TARGET_POSE = [
    0.434,
    0.456,
    0.495,
    57.4,
    -179.4,
    -122.6,
]  # x, y, z, roll, pitch, yaw [m, deg]

robot = DoosanRobotController("192.168.1.30")


def ik_callback():
    global is_init_ik, robot, solver, INIT_JOINT, TARGET_POSE

    while True:
        if not is_init_ik:
            continue

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

    # Robot On
    robot.connect()
    time.sleep(0.1)
    robot.servo_on()
    time.sleep(3.0)

    robot.movej(INIT_JOINT, 3.0)  # 3 sec moving

    # RT Mode Start
    robot.start_rt(INIT_JOINT[0:6])
    time.sleep(3.0)

    # ----------------------------------------------
    # 초기 조인트 값 설정
    # 기본적으로 7개의 조인트값을 입력함 7축 로봇 URDF가 아니라면 마지막 값은 쓰레기값으로 처리됨
    # ----------------------------------------------
    if not solver.init():
        print("Error: PyIk 초기화 실패")
        return

    solver.set_joint(INIT_JOINT, use_deg=True)
    solver.set_joint_range_limit(2, -1, -160)
    # 3번째 조인트의 동작 범위 설정
    is_init_ik = True

    th = threading.Thread(target=ik_callback)
    th.start()

    # Robot Control Loop
    while True:

        res = solver.get_current_joint(use_deg=True)
        res[5] = res[5] + math.sin(10 * time.time())

        print(res[0:6])
        robot.movej_rt(res[0:6], 0.001)
        time.sleep(0.001)


if __name__ == "__main__":
    main()
