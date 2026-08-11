import json
import multiprocessing as mp
import os
import socket
import time
import cv2
import numpy as np
import openvino as ov

from camera.gemini336 import SHM_NAME, CameraBuffer, camera_runner
from tracker.tracker import KalmanFilter3D, draw_3d_tf_axis

INT8_MODEL_PATH = "model_int8.xml"
CONF_THRES = 0.8
ESC_KEY = 27

# 카메라 내적 (Intrinsic Parameters)
FX, FY = 693.3102, 693.4061
CX, CY = 639.6599, 365.0724

# 로봇/제어 프로세스 통신용 UDP 설정
CONTROL_IP = "127.0.0.1"
CONTROL_PORT = 5005

# NMS-Free 디코더용 Anchor 사전 계산
CONF = [(8, 48, 80), (16, 24, 40), (32, 12, 20)]
REG_MAX = 16


def generate_anchors():
    anchors = []
    strides = []
    for stride, h, w in CONF:
        grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        grid = np.stack((grid_x, grid_y), axis=-1).reshape(-1, 2) + 0.5
        anchors.append(grid)
        strides.append(np.full((grid.shape[0], 1), stride))

    anchors = np.concatenate(anchors, axis=0)  # (5040, 2)
    strides = np.concatenate(strides, axis=0)  # (5040, 1)
    return anchors, strides


ANCHORS, STRIDES = generate_anchors()
kf_tracker = KalmanFilter3D(dt=1 / 30.0)


# ---------------------------------------------------------
# 2. 전처리 & 후처리 (NumPy 기반)
# ---------------------------------------------------------
def preprocess_image(color_rgb):
    """RGB Image -> Resize(360, 640) -> Pad(384, 640) -> Float32 [0, 1]"""
    resized = cv2.resize(color_rgb, (640, 360), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(
        resized, 12, 12, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    input_tensor = padded.astype(np.float32) / 255.0
    input_tensor = np.transpose(input_tensor, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(input_tensor)


def decode_bboxes(reg_pred):
    """DFL Softmax 및 Anchor 기반 BBox 좌표 복원"""
    if reg_pred.ndim == 3:
        reg_pred = reg_pred[0]

    reg_pred = reg_pred.reshape(-1, 4, REG_MAX)
    e_x = np.exp(reg_pred - np.max(reg_pred, axis=-1, keepdims=True))
    softmax_reg = e_x / np.sum(e_x, axis=-1, keepdims=True)

    weights = np.arange(REG_MAX, dtype=np.float32)
    dist = np.sum(softmax_reg * weights, axis=-1)

    x1 = (ANCHORS[:, 0] - dist[:, 0]) * STRIDES[:, 0]
    y1 = (ANCHORS[:, 1] - dist[:, 1]) * STRIDES[:, 0]
    x2 = (ANCHORS[:, 0] + dist[:, 2]) * STRIDES[:, 0]
    y2 = (ANCHORS[:, 1] + dist[:, 3]) * STRIDES[:, 0]

    return np.stack([x1, y1, x2, y2], axis=-1)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------
# 3. 동기식 메인 파이프라인
# ---------------------------------------------------------
def main():
    mp.set_start_method("spawn", force=True)
    stop_signal = mp.Event()

    # 카메라 수신 프로세스용 Lockless Buffer 생성 및 시작
    buffer = CameraBuffer(name=SHM_NAME, is_owner=True)
    camera_process = mp.Process(target=camera_runner, args=(SHM_NAME, stop_signal))
    camera_process.start()

    core = ov.Core()
    device_name = "GPU" if "GPU" in core.available_devices else "CPU"
    print(f"📦 OpenVINO INT8 모델 로드 중... [디바이스: {device_name}]")

    if not os.path.exists(INT8_MODEL_PATH):
        print(f"❌ [에러] 모델 파일({INT8_MODEL_PATH})이 존재하지 않습니다.")
        stop_signal.set()
        camera_process.join()
        buffer.close()
        return 1

    ov_model = core.read_model(INT8_MODEL_PATH)
    compiled_model = core.compile_model(ov_model, device_name)

    # 출력 레이어 바인딩 (순서 고정)
    output_0 = compiled_model.output(0)
    output_1 = compiled_model.output(1)

    prev_ts = 0.0
    prev_time = time.time()
    fps = 0.0

    print("🚀 동기식 추론 및 3D 추적 루프가 시작되었습니다.")

    try:
        while True:
            # 1. 공유 메모리에서 최신 카메라 프레임 가져오기
            current_frame = buffer.read_latest_frame()

            if current_frame.timestamp == prev_ts or current_frame.timestamp == 0.0:
                time.sleep(0.001)  # 새 프레임 대기 Polling
                continue

            prev_ts = current_frame.timestamp
            color = current_frame.color
            depth = current_frame.depth
            view = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

            # 2. 전처리
            input_tensor = preprocess_image(color)

            # 3. [동기식 추론] 결과가 반환될 때까지 블로킹 대기
            results = compiled_model({0: input_tensor})

            cls_pred = results[output_0]
            reg_pred = results[output_1]

            # Output Tensor Shape 매칭 (cls: scores, reg: bboxes)
            if cls_pred.shape[-1] != 1 and cls_pred.shape[-1] != 5040:
                cls_pred, reg_pred = reg_pred, cls_pred

            # 4. 후처리 및 디코딩 (즉시 계산)
            scores = sigmoid(cls_pred.squeeze())
            decoded_boxes = decode_bboxes(reg_pred)

            keep = scores > CONF_THRES
            final_boxes = decoded_boxes[keep]
            final_scores = scores[keep]

            measured_3d = None
            if len(final_boxes) > 0 and len(final_scores) > 0:
                # 좌표 스케일 복원 (Rescale / Pad 오프셋)
                final_boxes[:, [0, 2]] = final_boxes[:, [0, 2]] * 2
                final_boxes[:, [1, 3]] = (final_boxes[:, [1, 3]] - 12) * 2

                best_idx = np.argmax(final_scores)
                best_box = final_boxes[best_idx].tolist()

                # Depth 기반 3D Centroid 측정
                measured_3d = kf_tracker.compute_3d_centroid(
                    depth, best_box, FX, FY, CX, CY
                )

                x1, y1, x2, y2 = map(int, best_box)
                cv2.rectangle(view, (x1, y1), (x2, y2), (80, 80, 80), 1)

            # 5. 3D Kalman Filter 업데이트
            state, is_updated = kf_tracker.update(measured_3d)

            # 6. 제어 프로세스로 추적 결과 전송 (UDP)
            if kf_tracker.is_initialized:
                xc, yc, zc, vx, vy, vz = state

                control_packet = {
                    "timestamp": float(current_frame.timestamp),
                    "pos_mm": [float(xc), float(yc), float(zc)],
                    "vel_mms": [float(vx), float(vy), float(vz)],
                    "is_valid": bool(is_updated),
                }
                sock.sendto(
                    json.dumps(control_packet).encode("utf-8"),
                    (CONTROL_IP, CONTROL_PORT),
                )

                # 3D TF 축 시각화
                draw_3d_tf_axis(
                    img=view,
                    center_3d=(xc, yc, zc),
                    fx=FX,
                    fy=FY,
                    cx=CX,
                    cy=CY,
                    axis_length=80.0,
                    thickness=2,
                )

                # 화면 텍스트 출력
                pos_text = f"TF [X:{xc:.0f}, Y:{yc:.0f}, Z:{zc:.0f}] mm"
                vel_text = f"V [Vx:{vx:.0f}, Vy:{vy:.0f}, Vz:{vz:.0f}] mm/s"
                cv2.putText(
                    view,
                    pos_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    view,
                    vel_text,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            # FPS 계산 및 디스플레이
            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / (curr_time - prev_time + 1e-6))
            prev_time = curr_time

            cv2.putText(
                view,
                f"FPS: {fps:.1f}",
                (15, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("color", view)

            if cv2.waitKey(1) in (ord("q"), ESC_KEY):
                break

    finally:
        cv2.destroyAllWindows()
        sock.close()
        stop_signal.set()
        camera_process.join(timeout=3)
        if camera_process.is_alive():
            camera_process.terminate()
        buffer.close()
        print("[메인] 시스템이 안전하게 종료되었습니다.")


if __name__ == "__main__":
    raise SystemExit(main())