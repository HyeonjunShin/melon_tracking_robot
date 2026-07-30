import os
import json
import time
from pathlib import Path
import numpy as np
import cv2
import multiprocessing as mp
import openvino as ov

from lib.tracker.tracker import KalmanFilter3D, draw_3d_tf_axis
from lib.camera.gemini336 import runner, LocklessBuffer, SHM_NAME

# OpenVINO 모델 설정
INT8_MODEL_PATH = "model_int8.xml"
CONF_THRES = 0.45

# 카메라 및 추적 파라미터
CALIBRATION_JSON_PATH = "gemini336_hand_eye_settings.json"
ESC_KEY = 27

FX, FY = 693.3102, 693.4061
CX, CY = 639.6599, 365.0724
kf_tracker = KalmanFilter3D(dt=1 / 30.0)

# NMS-Free 디코더용 Anchor 및 Anchor-grid 사전 계산 (5040 개)
# [(stride, h, w)]
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


# ---------------------------------------------------------
# 1. NumPy 기반 전처리 (Torchvision Transform 대체)
# ---------------------------------------------------------
def preprocess_image(color_rgb):
    """(H, W, C) RGB -> Resize(360, 640) -> Pad(0, 12, 0, 12) -> (384, 640, C) -> (1, 3, 384, 640) float32 [0, 1]"""
    # Resize: (384-24, 640) = (360, 640)
    resized = cv2.resize(color_rgb, (640, 360), interpolation=cv2.INTER_LINEAR)

    # Pad: top=12, bottom=12, left=0, right=0 -> (384, 640, 3)
    padded = cv2.copyMakeBorder(
        resized, 12, 12, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    # ToTensor & Normalize (0~1 scale)
    input_tensor = padded.astype(np.float32) / 255.0
    input_tensor = np.transpose(input_tensor, (2, 0, 1))[None, ...]  # (1, 3, 384, 640)
    return np.ascontiguousarray(input_tensor)


# ---------------------------------------------------------
# 2. NumPy 기반 BBox 디코더
# ---------------------------------------------------------
def decode_bboxes(reg_pred):
    """
    reg_pred: (1, 5040, 64) 또는 (5040, 64) shape
    """
    if reg_pred.ndim == 3:
        reg_pred = reg_pred[0]

    # DFL Softmax
    reg_pred = reg_pred.reshape(-1, 4, REG_MAX)
    # Softmax over last dim
    e_x = np.exp(reg_pred - np.max(reg_pred, axis=-1, keepdims=True))
    softmax_reg = e_x / np.sum(e_x, axis=-1, keepdims=True)

    weights = np.arange(REG_MAX, dtype=np.float32)
    dist = np.sum(softmax_reg * weights, axis=-1)  # (5040, 4) [left, top, right, bottom]

    # Distance to BBox (x1, y1, x2, y2)
    x1 = (ANCHORS[:, 0] - dist[:, 0]) * STRIDES[:, 0]
    y1 = (ANCHORS[:, 1] - dist[:, 1]) * STRIDES[:, 0]
    x2 = (ANCHORS[:, 0] + dist[:, 2]) * STRIDES[:, 0]
    y2 = (ANCHORS[:, 1] + dist[:, 3]) * STRIDES[:, 0]

    decoded_boxes = np.stack([x1, y1, x2, y2], axis=-1)
    return decoded_boxes


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------
# 3. 실시간 OpenVINO 메인 추론 엔진
# ---------------------------------------------------------
def main():
    mp.set_start_method("spawn", force=True)
    stop_signal = mp.Event()

    buffer = LocklessBuffer(name=SHM_NAME, is_owner=True)

    p = mp.Process(target=runner, args=(SHM_NAME, stop_signal))
    p.start()

    # 1. OpenVINO 디바이스 및 코어 초기화
    core = ov.Core()
    available_devices = core.available_devices
    device_name = "GPU" if "GPU" in available_devices else "CPU"
    print(
        f"\n📦 OpenVINO INT8 모델({INT8_MODEL_PATH}) 로드 중... [디바이스: {device_name}]"
    )

    if not os.path.exists(INT8_MODEL_PATH):
        print(f"❌ [에러] OpenVINO XML 모델 파일({INT8_MODEL_PATH})이 없습니다.")
        stop_signal.set()
        p.join()
        buffer.close()
        return 1

    ov_model = core.read_model(INT8_MODEL_PATH)
    compiled_model = core.compile_model(
        ov_model, device_name, {"PERFORMANCE_HINT": "THROUGHPUT"}
    )

    infer_queue = ov.AsyncInferQueue(compiled_model)
    print(f"-> [OpenVINO INT8 AsyncInferQueue 가동 완료: {len(infer_queue)} 스레드/슬롯]")

    # OpenVINO 추론 결과 보관용 공유 변수 구조
    latest_results = {"boxes": np.empty((0, 4)), "scores": np.empty((0,))}

    def completion_callback(infer_request, userdata):
        # 모델의 출력 노드 순서에 따른 결과 추출 (cls, reg)
        cls_pred = infer_request.get_output_tensor(0).data
        reg_pred = infer_request.get_output_tensor(1).data

        # shape 자동 매칭 (cls_pred: scores, reg_pred: bboxes)
        if cls_pred.shape[-1] != 1 and cls_pred.shape[-1] != 5040:
            cls_pred, reg_pred = reg_pred, cls_pred

        scores = sigmoid(cls_pred.squeeze())
        decoded_boxes = decode_bboxes(reg_pred)

        keep = scores > CONF_THRES
        latest_results["boxes"] = decoded_boxes[keep]
        latest_results["scores"] = scores[keep]

    infer_queue.set_callback(completion_callback)

    # Warm-up
    dummy_input_np = np.zeros((1, 3, 384, 640), dtype=np.float32)
    for _ in range(len(infer_queue) * 2):
        infer_queue.start_async({0: dummy_input_np})
    infer_queue.wait_all()

    prev_ts = 0.0
    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            current_frame = buffer.read_latest_frame()

            if current_frame.timestamp == prev_ts or current_frame.timestamp == 0.0:
                time.sleep(0.001)
                continue

            prev_ts = current_frame.timestamp

            color = current_frame.color
            depth = current_frame.depth
            view = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

            # 전처리 및 비동기 추론 요청
            input_tensor = preprocess_image(color)
            infer_queue.start_async({0: input_tensor})

            # 최신 바운딩 박스 결과 취득
            final_boxes = latest_results["boxes"].copy()
            final_scores = latest_results["scores"].copy()

            measured_3d = None
            if len(final_boxes) > 0:
                # 좌표 복원 (Rescale / Pad 제거 오프셋)
                final_boxes[:, [0, 2]] = final_boxes[:, [0, 2]] * 2
                final_boxes[:, [1, 3]] = (final_boxes[:, [1, 3]] - 12) * 2

                best_idx = np.argmax(final_scores)
                best_box = final_boxes[best_idx].tolist()

                # 3D Centroid (Xc, Yc, Zc) 측정
                measured_3d = kf_tracker.compute_3d_centroid(
                    depth, best_box, FX, FY, CX, CY
                )

                x1, y1, x2, y2 = map(int, best_box)
                cv2.rectangle(view, (x1, y1), (x2, y2), (80, 80, 80), 1)

            state, is_updated = kf_tracker.update(measured_3d)

            if kf_tracker.is_initialized:
                xc, yc, zc, vx, vy, vz = state

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
            key = cv2.waitKey(1)

            if key == ord("q") or key == ESC_KEY:
                break

    finally:
        infer_queue.wait_all()
        cv2.destroyAllWindows()
        stop_signal.set()
        p.join(timeout=3)
        if p.is_alive():
            p.terminate()
        buffer.close()
        print("[메인] 시스템이 안전하게 종료되었습니다.")


if __name__ == "__main__":
    raise SystemExit(main())