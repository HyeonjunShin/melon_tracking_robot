import numpy as np
import cv2


def draw_3d_tf_axis(
    img,
    center_3d,
    fx,
    fy,
    cx,
    cy,
    axis_length=80.0,
    rotation_matrix=None,
    thickness=2,
):
    xc, yc, zc = center_3d
    if zc <= 0:
        return

    # 1. 3D 축의 원점 및 X, Y, Z 축 끝점 정의 (기본값: 카메라 좌표계와 동일 방향)
    if rotation_matrix is None:
        R = np.eye(3, dtype=np.float32)
    else:
        R = rotation_matrix

    # 원점과 각 축 방향 벡터 (mm 단위)
    origin_3d = np.array([xc, yc, zc], dtype=np.float32)
    x_axis_3d = origin_3d + R @ np.array([axis_length, 0, 0], dtype=np.float32)
    y_axis_3d = origin_3d + R @ np.array([0, axis_length, 0], dtype=np.float32)
    z_axis_3d = origin_3d + R @ np.array([0, 0, axis_length], dtype=np.float32)

    # 2. 3D Points -> 2D Image Pixels Projection (u = fx * X/Z + cx, v = fy * Y/Z + cy)
    pts_3d = np.vstack([origin_3d, x_axis_3d, y_axis_3d, z_axis_3d])

    u = (pts_3d[:, 0] * fx / pts_3d[:, 2]) + cx
    v = (pts_3d[:, 1] * fy / pts_3d[:, 2]) + cy
    pts_2d = np.column_stack([u, v]).astype(int)

    p_orig = tuple(pts_2d[0])
    p_x = tuple(pts_2d[1])
    p_y = tuple(pts_2d[2])
    p_z = tuple(pts_2d[3])

    # 3. OpenCV 화면에 TF 축 그리기 (X: Red, Y: Green, Z: Blue)
    cv2.line(img, p_orig, p_x, (0, 0, 255), thickness, cv2.LINE_AA)  # X-axis (Red)
    cv2.line(img, p_orig, p_y, (0, 255, 0), thickness, cv2.LINE_AA)  # Y-axis (Green)
    cv2.line(img, p_orig, p_z, (255, 0, 0), thickness, cv2.LINE_AA)  # Z-axis (Blue)

    # 중심점 원 표시
    cv2.circle(img, p_orig, 4, (255, 255, 255), -1)

    # 축 라벨 표시
    cv2.putText(
        img,
        "X",
        p_x,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "Y",
        p_y,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "Z",
        p_z,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 0, 0),
        1,
        cv2.LINE_AA,
    )


class CentroidTracker3D:
    """
    3D Centroid (X, Y, Z) 전용 칼만 필터 트래커

    - State vector (6, 1): [x, y, z, vx, vy, vz]^T  (위치 및 속도)
    - Measurement (3, 1):  [x, y, z]^T              (측정 위치)
    """

    def __init__(self, dt=1 / 30.0):
        self.dt = dt
        self.is_initialized = False
        self.last_time = None

        # 상태 벡터 [x, y, z, vx, vy, vz]^T
        self.x = np.zeros((6, 1), dtype=np.float32)

        # 관측 행렬 (H): 위치 [x, y, z]만 관측
        self.H = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ],
            dtype=np.float32,
        )

        # 시스템(운동) 노이즈 및 센서 관측 노이즈 튜닝
        self.q_pos = 0.05  # 위치 시스템 노이즈
        self.q_vel = 0.5  # 속도 시스템 노이즈
        self.r_pos = 0.005  # Depth 센서 오차 분산

        self.R = np.eye(3, dtype=np.float32) * self.r_pos
        self.P = np.eye(6, dtype=np.float32) * 100.0

        self._update_matrices(self.dt)

    def _update_matrices(self, dt):
        """dt(시간 간격) 변경 시 상태 전이 행렬(F) 및 시스템 노이즈(Q) 업데이트"""
        # 등속도 모델 (Constant Velocity Model)
        self.F = np.array(
            [
                [1, 0, 0, dt, 0, 0],
                [0, 1, 0, 0, dt, 0],
                [0, 0, 1, 0, 0, dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

        self.Q = (
            np.diag([self.q_pos, self.q_pos, self.q_pos, self.q_vel, self.q_vel, self.q_vel]).astype(
                np.float32
            )
            * dt
        )

    def update(self, z_centroid=None, current_time=None):
        if current_time is not None:
            if self.last_time is not None:
                dt = current_time - self.last_time
                if dt > 0:
                    self._update_matrices(dt)
            self.last_time = current_time

        if not self.is_initialized:
            if z_centroid is not None:
                self.x[:3] = np.array(z_centroid, dtype=np.float32).reshape(3, 1)
                self.x[3:] = 0.0  # 초기 속도는 0
                self.is_initialized = True

            # [수정] (centroid_3d, velocity_3d, is_updated) 반환
            return self.x[:3].flatten().astype(np.float64), self.x[3:].flatten().astype(np.float64), False

        # Predict Step
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Update Step
        if z_centroid is not None:
            z_measured = np.array(z_centroid, dtype=np.float32).reshape(3, 1)
            y = z_measured - (self.H @ self.x)
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)

            self.x = self.x + (K @ y)
            I = np.eye(6, dtype=np.float32)
            self.P = (I - K @ self.H) @ self.P

            return self.x[:3].flatten().astype(np.float64), self.x[3:].flatten().astype(np.float64), True
        else:
            return self.x[:3].flatten().astype(np.float64), self.x[3:].flatten().astype(np.float64), False

    def predict_future_centroid(self, lead_time_sec=0.05):
        """
        시스템 Latency 보상을 위해 N초 후의 미래 3D Centroid 위치 예측
        """
        current_pos = self.x[:3].flatten()
        current_vel = self.x[3:].flatten()
        future_pos = current_pos + current_vel * lead_time_sec
        return future_pos.astype(np.float64)
