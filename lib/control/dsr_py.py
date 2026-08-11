import time
import doosan_robot_controller_py as drc


class DoosanRobotController(drc.DSR):
    def __init__(self, ip: str):
        super().__init__(ip)

    def connect(self):
        super().connect()

    def disconnect(self):
        super().disconnect()

    def servo_on(self):
        super().servo_on()

    def servo_off(self):
        super().servo_off()

    def stop(self):
        super().stop()

    def get_curr_joint_deg(self):
        return super().get_current_joint()

    def set_kp_gain(self, kp):
        super().set_kp_gain(kp)

    def set_kd_gain(self, kd):
        super().set_kd_gain(kd)

    def set_target_time(self, time):
        super().set_target_time(time)

    def movej(self, q, time):
        super().movej(q, time)  # deg

    def start_rt(self, q):
        return super().start_rt(q)

    def movej_rt(self, q, dt):
        super().movej_rt(q, dt)


if __name__ == "__main__":
    robot = DoosanRobotController("192.168.1.30")
    robot.connect()
    time.sleep(0.1)
    robot.servo_on()
    time.sleep(5)
    robot.disconnect()
