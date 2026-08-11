import numpy as np
import math
from typing import Union, List
import ik_solver_py


class PyIk:
    """
    pybind11로 바인딩된 ik_solver_py 모듈을
    파이썬 스타일(Pythonic)로 다루기 위한 래퍼 클래스입니다.
    """

    def __init__(self, urdf_path: str):
        """
        PyIk 인스턴스를 생성합니다.

        :param urdf_path: 로봇 URDF 파일의 경로
        """
        self._solver = ik_solver_py.IkSolver(urdf_path)
        self._is_initialized = False

    def init(self) -> bool:
        """
        로봇 모델 및 파라미터를 초기화합니다.
        """
        self._is_initialized = self._solver.init()
        return self._is_initialized

    def _ensure_initialized(self):
        if not self._is_initialized:
            raise RuntimeError(
                "IkSolver가 초기화되지 않았습니다. 먼저 init()을 호출하세요."
            )

    # --------------------------------------------------------------------------
    # 제어 함수 (Motion Controls)
    # --------------------------------------------------------------------------
    def movej(self, q: Union[np.ndarray, List[float]], use_deg: bool = False):
        """
        Joint Space 상에서 로봇을 목표 각도로 이동시킵니다.

        :param q: 목표 조인트 각도 (크기 7의 리스트 또는 NumPy 배열)
        :param use_deg: True일 경우 q를 Degree(도) 단위로 해석하여 내부적으로 Radian으로 변환
        """
        self._ensure_initialized()
        q_np = np.asarray(q, dtype=np.float64)

        if q_np.shape != (7,):
            raise ValueError(
                f"조인트 배열의 크기는 7이어야 합니다. 입력 크기: {q_np.shape}"
            )

        if use_deg:
            q_np = np.radians(q_np)

        self._solver.movej(q_np)

    def movel(self, target_matrix: np.ndarray):
        """
        Task Space(Cartesian) 상에서 로봇을 목표 포즈로 이동시킵니다.

        :param target_matrix: 4x4 동차 변환 행렬 (NumPy 2D 배열)
        """
        self._ensure_initialized()
        if target_matrix.shape != (4, 4):
            raise ValueError(
                f"목표 포즈 행렬은 4x4 크기여야 합니다. 입력 크기: {target_matrix.shape}"
            )

        self._solver.movel(target_matrix)

    # --------------------------------------------------------------------------
    # 상태 조회 함수 (Getters)
    # --------------------------------------------------------------------------
    def get_current_joint(self, use_deg: bool = False) -> np.ndarray:
        """현재 조인트의 각도를 반환합니다."""
        self._ensure_initialized()
        if use_deg:
            return self._solver.get_curr_joint_deg()
        return self._solver.get_curr_joint_rad()

    def get_current_joint_velocity(self, use_deg: bool = False) -> np.ndarray:
        """현재 조인트의 회전 속도를 반환합니다."""
        self._ensure_initialized()
        if use_deg:
            return self._solver.get_curr_joint_velocity_deg()
        return self._solver.get_curr_joint_velocity_rad()

    def get_current_tcp_speed(self) -> np.ndarray:
        """현재 TCP의 속도(6자유도 트위스트: 선속도 및 각속도)를 반환합니다."""
        self._ensure_initialized()
        return self._solver.get_curr_tcp_speed()

    def get_current_jacobian(self) -> np.ndarray:
        """현재 로봇의 6x7 자코비안(Jacobian) 행렬을 반환합니다."""
        self._ensure_initialized()
        return self._solver.get_curr_jacobian()

    def get_current_tcp_tf(self) -> np.ndarray:
        """현재 최종 단에 설정된 TCP의 4x4 포즈 행렬을 반환합니다."""
        self._ensure_initialized()
        return self._solver.get_curr_tcp_tf()

    def get_current_tf(self, index: int) -> np.ndarray:
        """특정 조인트/링크 인덱스의 4x4 포즈 행렬을 반환합니다."""
        self._ensure_initialized()
        return self._solver.get_curr_tf(index)

    def get_end_effector_offset(self) -> np.ndarray:
        """현재 설정된 엔드 이펙터(EE) 오프셋 행렬을 반환합니다."""
        self._ensure_initialized()
        return self._solver.get_end_effector_offset()

    def get_base_tf(self) -> np.ndarray:
        """로봇의 베이스 좌표계 변환 행렬을 반환합니다."""
        self._ensure_initialized()
        return self._solver.get_base_tf()

    # --------------------------------------------------------------------------
    # 설정 함수 (Setters)
    # --------------------------------------------------------------------------
    def set_joint(
        self, joint: Union[np.ndarray, List[float]], use_deg: bool = False
    ):
        """로봇의 현재 조인트 상태를 강제로 설정(동기화)합니다."""
        self._ensure_initialized()
        joint_np = np.asarray(joint, dtype=np.float64)
        if use_deg:
            joint_np = np.radians(joint_np)
        self._solver.set_joint(joint_np)

    def set_joint_range_limit(
        self, index: int, min_val: float, max_val: float, use_deg: bool = False
    ):
        """특정 조인트의 최소/최대 가동 범위를 제한합니다."""
        if use_deg:
            min_val = math.radians(min_val)
            max_val = math.radians(max_val)
        self._solver.set_joint_range_limit(index, min_val, max_val)

    def set_joint_velocity_limit(
        self, index: int, max_vel: float, use_deg: bool = False
    ):
        """특정 조인트의 최대 회전 속도를 제한합니다."""
        if use_deg:
            max_vel = math.radians(max_vel)
        self._solver.set_joint_velocity_limit(index, max_vel)

    def set_joint_velocity_limit_scale(self, scale: float):
        """전체 조인트 속도 제한에 가중치(Scale)를 적용합니다."""
        self._solver.set_joint_velocity_limit_scale(scale)

    def set_workspace_limits(
        self, x_range: tuple, y_range: tuple, z_range: tuple
    ):
        """
        로봇 TCP가 움직일 수 있는 3차원 작업 영역(Workspace)을 제한합니다.

        :param x_range: (min_x, max_x) 형태의 튜플 [m]
        :param y_range: (min_y, max_y) 형태의 튜플 [m]
        :param z_range: (min_z, max_z) 형태의 튜플 [m]
        """
        self._solver.set_workspace_limitX(x_range[0], x_range[1])
        self._solver.set_workspace_limitY(y_range[0], y_range[1])
        self._solver.set_workspace_limitZ(z_range[0], z_range[1])

    def set_tcp_max_speed(self, max_speed: float):
        """TCP의 최대 선속도를 제한합니다 [m/s]."""
        self._solver.set_tcp_max_speed(max_speed)

    def set_target_elbow_angle(self, angle: float, use_deg: bool = False):
        """7자유도 로봇 제어를 위한 팔꿈치(Elbow) 목표 각도를 설정합니다."""
        if use_deg:
            angle = math.radians(angle)
        self._solver.set_target_elbow_angle(angle)

    def set_elbow_control_mode(self, enable: bool):
        """팔꿈치 제어 모드를 활성화 또는 비활성화합니다."""
        self._solver.set_elbow_control_mode(enable)

    def set_end_effector_offset(self, tf_matrix: np.ndarray):
        """4x4 NumPy 행렬을 입력받아 엔드 이펙터 오프셋을 설정합니다."""
        if tf_matrix.shape != (4, 4):
            raise ValueError("엔드 이펙터 행렬은 4x4 크기여야 합니다.")
        self._solver.set_end_effector_offset(tf_matrix)

    def set_base_transform(self, tf_matrix: np.ndarray):
        """4x4 NumPy 행렬을 입력받아 베이스 좌표계를 변환합니다."""
        if tf_matrix.shape != (4, 4):
            raise ValueError("베이스 변환 행렬은 4x4 크기여야 합니다.")
        self._solver.set_base_transform(tf_matrix)

    # --------------------------------------------------------------------------
    # 정적 메서드 유틸리티 (Static Utilities)
    # --------------------------------------------------------------------------
    @staticmethod
    def make_tf(
        x: float,
        y: float,
        z: float,
        roll: float,
        pitch: float,
        yaw: float,
        use_deg: bool = False,
    ) -> np.ndarray:
        """
        위치(XYZ)와 오일러 각(RPY) 정보를 조합하여 4x4 동차 변환 행렬을 생성합니다.

        :param use_deg: True일 경우 roll, pitch, yaw 단위를 Degree로 간주하여 내부적 변환 수행
        """
        if use_deg:
            roll = math.radians(roll)
            pitch = math.radians(pitch)
            yaw = math.radians(yaw)
        return ik_solver_py.make_tf(x, y, z, roll, pitch, yaw)


from typing import Any, Dict, Tuple
from multiprocessing import shared_memory, resource_tracker
import atexit


class SharedMemory:
    def __init__(self, name: str, fields_config: Dict[str, Tuple[int, Any]]):
        self.name = name
        self.fields = {}
        offset = 0
        for f_name, (count, dtype) in fields_config.items():
            item_size = 1 if dtype == str else np.dtype(dtype).itemsize
            byte_size = count * item_size
            self.fields[f_name] = {
                "offset": offset,
                "count": count,
                "dtype": dtype,
                "byte_size": byte_size,
            }
            offset += byte_size
        self.total_size = offset

        try:
            self.shm = shared_memory.SharedMemory(
                name=name, create=True, size=self.total_size
            )
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(name=name)

        try:
            resource_tracker.unregister(self.shm._name, "shared_memory")
        except:
            pass
        atexit.register(self.close)

    def set(self, field_name: str, value: Any) -> None:
        f = self.fields[field_name]
        if f["dtype"] == str:
            encoded = str(value).encode("utf-8")[: f["byte_size"]]
            self.shm.buf[f["offset"] : f["offset"] + len(encoded)] = encoded
        else:
            arr = np.ndarray(
                (f["count"],),
                dtype=f["dtype"],
                buffer=self.shm.buf,
                offset=f["offset"],
            )
            arr[:] = value

    def close(self) -> None:
        if hasattr(self, "shm"):
            self.shm.close()


def main():
    # ----------------------------------------------
    # 블렌더 시각화용 공유 메모리 생성
    # ----------------------------------------------
    fields_config = {"joint": (7, np.float32), "status": (20, str)}
    shm = SharedMemory(name="movej", fields_config=fields_config)

    # ----------------------------------------------
    # IK solver 생성
    # ----------------------------------------------
    URDF = "./data/robot/urdf/doosan_m1013.urdf"
    solver = PyIk(URDF)

    if not solver.init():
        print("Error: PyIk 초기화 실패")
        return

    # ----------------------------------------------
    # 초기 조인트 값 설정
    # 기본적으로 7개의 조인트값을 입력함 7축 로봇 URDF가 아니라면 마지막 값은 쓰레기값으로 처리됨
    # ----------------------------------------------
    solver.set_joint([0, 0, 0, 0, 0, 0, 0], use_deg=True)
    # solver.set_joint_range_limit(2, -1, -160); # 3번째 조인트의 동작 범위 설정

    # ----------------------------------------------
    # 제어 루프 (비동기를 추천)
    # ----------------------------------------------
    import time

    target_q_rad = [0.0] * 7
    t = 0.0
    while True:
        # movej
        target_q_rad[2] = math.sin(t)  # 3번째 조인트 sin파 모션 생성
        solver.movej(target_q_rad, use_deg=False)

        # movel
        # target_pose = [0.4, 0.4, 0.4, 0.0, 90.0, 0.0]
        #
        # target_matrix = PyIk.make_tf(
        #     target_pose[0], target_pose[1], target_pose[2], # x, y, z [m]
        #     target_pose[3], target_pose[4], target_pose[5], # roll, pitch, yaw [rad]
        #     use_deg=False
        # )
        # solver.movel(target_matrix) # 4x4 matrix가 입력임

        # 조인트 값 출력 (소수점 2자리 정렬)
        curr_q_deg = solver.get_current_joint(use_deg=True)
        q_str = ", ".join([f"{q:>7.2f}" for q in curr_q_deg])
        print(f"▶ Joint (deg): [{q_str}]")

        # tcp 값 출력 (4x4 행렬 소수점 2자리 정렬)
        curr_tf = solver.get_current_tcp_tf()
        print("▶ TCP Transform Matrix:")
        for row in curr_tf:
            row_str = ", ".join([f"{val:>8.2f}" for val in row])
            print(f"  [{row_str}]")
        print("-" * 50)  # 반복 출력 시 구분을 위한 점선

        # 공유 메리리에 데이터 쓰기
        shm.set("joint", curr_q_deg.astype(np.float32))
        shm.set("status", "run")

        # 루프 끝
        t += 0.01
        time.sleep(0.01)


if __name__ == "__main__":
    main()
