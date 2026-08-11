import numpy as np
import torchvision.transforms.v2 as v2
from scipy.spatial.transform import Rotation as R


def draw_3d_axis(img, pose_7d, fx, fy, cx, cy, axis_length=0.1):
    """
    3D Pose (X, Y, Z, qx, qy, qz, qw)를 입력받아 RGB 3D 좌표축을 이미지에 렌더링
    """
    # 1. 값이 0으로 채워진 경우 스킵
    if np.all(pose_7d == 0):
        return

    tvec = pose_7d[:3].copy()  # [X, Y, Z]
    quat = pose_7d[3:].copy()  # [qx, qy, qz, qw]

    # [디버그/보정] Z값이 10 이상이면 mm 단위로 판단하여 m(미터) 단위로 스케일 변환 및 축 길이 자동 조절
    if tvec[2] > 10.0:
        tvec /= 1000.0  # mm -> m 변환

    # Quaternion 검증 (Norm이 0에 가까우면 기본 단위 쿼터니언으로 대체)
    quat_norm = np.linalg.norm(quat)
    if quat_norm < 1e-6:
        quat = np.array([0.0, 0.0, 0.0, 1.0])
    else:
        quat = quat / quat_norm

    # Quaternion -> Rotation Matrix
    r_matrix = R.from_quat(quat).as_matrix()

    # 3D 축 생성 (Centroid 기준 X, Y, Z 방향)
    axes_3d = np.array(
        [
            [0, 0, 0],  # Origin
            [axis_length, 0, 0],  # X-axis (Red)
            [0, axis_length, 0],  # Y-axis (Green)
            [0, 0, axis_length],  # Z-axis (Blue)
        ],
        dtype=np.float64,
    )

    # 3D 좌표 변환 (Rotated & Translated Points)
    axes_transformed = (r_matrix @ axes_3d.T).T + tvec

    # Z 좌표가 0 이하(카메라 뒤쪽)인 경우 스킵
    if np.any(axes_transformed[:, 2] <= 0):
        # print(f"[DEBUG] Z <= 0 발생: {axes_transformed[:, 2]}") # 필요시 주석 해제
        return

    # Pinhole Camera Model Projection (3D -> 2D Pixel)
    u = (axes_transformed[:, 0] * fx / axes_transformed[:, 2]) + cx
    v = (axes_transformed[:, 1] * fy / axes_transformed[:, 2]) + cy
    pts_2d = np.stack([u, v], axis=-1).astype(int)

    origin = tuple(pts_2d[0])
    pt_x = tuple(pts_2d[1])
    pt_y = tuple(pts_2d[2])
    pt_z = tuple(pts_2d[3])

    # [디버그] 중심 원 그리기 (위치 잡히는지 확인용)
    cv2.circle(img, origin, 5, (0, 255, 255), -1)

    # 3D 좌표축 선 그리기 (두께 3으로 강화)
    cv2.line(img, origin, pt_x, (0, 0, 255), 3)  # X-axis: RED
    cv2.line(img, origin, pt_y, (0, 255, 0), 3)  # Y-axis: GREEN
    cv2.line(img, origin, pt_z, (255, 0, 0), 3)  # Z-axis: BLUE  # Z-axis: BLUE
