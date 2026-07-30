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


class KalmanFilter3D:

    def __init__(self, dt=1 / 30.0):
        self.dt = dt
        self.is_initialized = False

        # State vector: [x, y, z, vx, vy, vz]^T (6, 1)
        self.x = np.zeros((6, 1), dtype=np.float32)

        # 1. State Transition Matrix (F) - 등속도 모델
        # x_k = x_{k-1} + vx * dt
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

        # 2. Measurement Matrix (H) - [x, y, z] 3개 위치만 관측
        self.H = np.array(
            [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]],
            dtype=np.float32,
        )

        # 3. Process Noise Covariance (Q) - 시스템(운동) 모델 노이즈
        # 값이 클수록 관측치(센서)를 더 신뢰하고 반응이 빨라지지만 떨림이 커짐
        q_pos = 100.0  # 위치 시스템 노이즈
        q_vel = 1000.0  # 속도 시스템 노이즈
        self.Q = (
            np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel]).astype(np.float32) * dt
        )

        # 4. Measurement Noise Covariance (R) - Depth 센서 오차 노이즈 (mm 단위)
        # 값이 클수록 Depth 센서의 튀는 노이즈를 부드럽게 Smoothing 함
        r_pos = 4.0  # 약 5mm 수준의 관측 오차 노이즈
        self.R = np.eye(3, dtype=np.float32) * r_pos

        # 5. Estimate Error Covariance (P) - 상태 추정 오차 공분산
        self.P = np.eye(6, dtype=np.float32) * 100.0

    def compute_3d_centroid(self, depth_img, box, fx, fy, cx, cy, stride=2):
        """2D BBox 영역의 Depth 데이터를 3D Centroid (Xc, Yc, Zc)로 변환 (mm 단위)"""
        x1, y1, x2, y2 = map(int, box)

        h_img, w_img = depth_img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        roi_depth = depth_img[y1:y2:stride, x1:x2:stride].squeeze()

        # 유효 Depth 마스킹 (15mm ~ 2000mm)
        valid_mask = (roi_depth > 15) & (roi_depth < 2000)
        if np.sum(valid_mask) < 5:
            return None

        v_grid, u_grid = np.mgrid[y1:y2:stride, x1:x2:stride]

        u_valid = u_grid[valid_mask]
        v_valid = v_grid[valid_mask]
        z_valid = roi_depth[valid_mask].astype(np.float32)

        x_valid = (u_valid - cx) * z_valid / fx
        y_valid = (v_valid - cy) * z_valid / fy

        xc = np.median(x_valid)
        yc = np.median(y_valid)
        zc = np.median(z_valid)

        return np.array([[xc], [yc], [zc]], dtype=np.float32)

    def update(self, z_measured=None):
        """Kalman Filter Predict & Update Step"""
        # ----------------------------------------------------
        # 1. Predict Step (예측 단계)
        # ----------------------------------------------------
        if not self.is_initialized:
            if z_measured is not None:
                self.x[:3] = z_measured
                self.x[3:] = 0.0
                self.is_initialized = True
            return self.x.flatten(), False

        # x_{k|k-1} = F * x_{k-1|k-1}
        self.x = self.F @ self.x
        # P_{k|k-1} = F * P_{k-1|k-1} * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

        # ----------------------------------------------------
        # 2. Update Step (측정치 반영 단계)
        # ----------------------------------------------------
        if z_measured is not None:
            # Innovation (측정 잔차): y = z - H * x
            y = z_measured - (self.H @ self.x)

            # Innovation Covariance: S = H * P * H^T + R
            S = self.H @ self.P @ self.H.T + self.R

            # Kalman Gain: K = P * H^T * S^-1
            K = self.P @ self.H.T @ np.linalg.inv(S)

            # Updated State: x = x + K * y
            self.x = self.x + (K @ y)

            # Updated Covariance: P = (I - K * H) * P
            I = np.eye(6, dtype=np.float32)
            self.P = (I - K @ self.H) @ self.P

            return self.x.flatten(), True
        else:
            # 2D 인식이 실패해도 이전 속도 기반 예측값(x_{k|k-1})을 그대로 유지
            return self.x.flatten(), False

    def predict_future(self, lead_time_sec=0.05):
        """
        lead_time_sec (초) 후의 미래 3D 위치를 등속도 모델 기반으로 예측합니다.
        (예: 0.05초 = 50ms 지연시간 보상)
        """
        current_pos = self.x[:3].flatten()
        current_vel = self.x[3:].flatten()

        # P_future = P_current + V * t_lead
        future_pos = current_pos + current_vel * lead_time_sec
        return future_pos
