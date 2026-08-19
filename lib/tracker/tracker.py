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
    def __init__(self, dt=1 / 30.0):
        self.dt = dt
        self.is_initialized = False
        self.last_time = None

        self.x = np.zeros((6, 1), dtype=np.float32)

        self.H = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ],
            dtype=np.float32,
        )

        # -------------------------------------------------------------
        # 💡 [반응성 최적화 튜닝]
        # -------------------------------------------------------------
        # 1. r_pos: 0.08 -> 0.008로 낮추어 측정값 반영 속도를 획기적으로 올립니다.
        self.r_pos = 0.001

        # 2. q_acc: 0.05 -> 0.8로 올려 급격한 가속도 변화를 빠르게 추종합니다.
        self.q_acc = 2.0

        self.R = np.eye(3, dtype=np.float32) * self.r_pos
        self.P = np.eye(6, dtype=np.float32) * 1.0

        self._update_matrices(self.dt)

    def _update_matrices(self, dt):
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

        q_p = 0.25 * (dt**4) * self.q_acc
        q_v = (dt**2) * self.q_acc

        self.Q = np.diag([q_p, q_p, q_p, q_v, q_v, q_v]).astype(np.float32)

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
                self.x[3:] = 0.0
                self.is_initialized = True

            return (
                self.x[:3].flatten().astype(np.float64),
                self.x[3:].flatten().astype(np.float64),
                False,
            )

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

            MAX_VEL = 2.5
            self.x[3:] = np.clip(self.x[3:], -MAX_VEL, MAX_VEL)

            # =========================================================
            # 💡 [수정 구역] 정지 상태 감지 (Zero-Lock) & 조건부 Lead Time
            # =========================================================
            # 1. 측정된 위치 변화량(Distance) 계산
            dist = np.linalg.norm(z_measured - self.x[:3])

            # 2. 이동량이 5mm 미만이면 정지 상태로 판단하여 속도를 0으로 고정
            STOP_THRESHOLD = 0.005  # 5mm
            if dist < STOP_THRESHOLD:
                self.x[3:] = 0.0  # 속도 누적 차단 (Drift 방지)

            # 3. 속도가 3cm/s 이상일 때만 lead_time 적용 (정지 시 lead_time=0)
            current_vel_norm = np.linalg.norm(self.x[3:])
            if current_vel_norm > 0.03:
                effective_lead_time = 0.05
            else:
                effective_lead_time = 0.0

            # 4. 최종 반환 위치 계산
            pred_pos = self.x[:3].flatten() + self.x[3:].flatten() * effective_lead_time
            # =========================================================

            return (
                pred_pos.astype(np.float64),
                self.x[3:].flatten().astype(np.float64),
                True,
            )
        else:
            self.x[3:] *= 0.90

            return (
                self.x[:3].flatten().astype(np.float64),
                self.x[3:].flatten().astype(np.float64),
                False,
            )
