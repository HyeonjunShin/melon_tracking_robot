import os
import json
import time
from pathlib import Path
import torch
import numpy as np
import cv2
import multiprocessing as mp
import torchvision.transforms.v2 as v2

from lib.detector.utiles import BBoxDecoder
from lib.detector.model import DetectionModel
from lib.tracker.tracker import KalmanFilter3D, draw_3d_tf_axis
from lib.camera.gemini336 import runner, LocklessBuffer, SHM_NAME

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONF_THRES = 0.45  # 주석 내용에 맞춰 0.45로 적용

# 모델 입력 해상도 (H, W) -> 384x640 규격 (5040 앵커 매칭)
MODEL_IMG_SIZE = (384, 640)
NUM_BBOXES = 5040
REG_MAX = 16
CONF = [(8, 48, 80), (16, 24, 40), (32, 12, 20)]
NUM_CLASSES = 1
MODEL_PATH = "./lib/detector/weight.pt"

CALIBRATION_JSON_PATH = "gemini336_hand_eye_settings.json"
ESC_KEY = 27

FX, FY = 693.3102, 693.4061
CX, CY = 639.6599, 365.0724
kf_tracker = KalmanFilter3D(dt=1 / 30.0)


def init_undistort_maps(json_path, w, h):
    if not os.path.exists(json_path):
        print(f"⚠️ 경고: '{json_path}' 파일이 없어 왜곡 보정을 건너뜁니다.")
        return None, None

    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
        calib_data = payload["calibration"]

    K = np.array(
        [
            [calib_data["rgb_intrinsic"]["fx"], 0, calib_data["rgb_intrinsic"]["cx"]],
            [0, calib_data["rgb_intrinsic"]["fy"], calib_data["rgb_intrinsic"]["cy"]],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    D = np.array(
        [
            calib_data["rgb_distortion"]["k1"],
            calib_data["rgb_distortion"]["k2"],
            calib_data["rgb_distortion"]["p1"],
            calib_data["rgb_distortion"]["p2"],
            calib_data["rgb_distortion"]["k3"],
            calib_data["rgb_distortion"]["k4"],
            calib_data["rgb_distortion"]["k5"],
            calib_data["rgb_distortion"]["k6"],
        ],
        dtype=np.float32,
    )

    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, new_K, (w, h), cv2.CV_32FC1)
    return mapx, mapy


transform = v2.Compose(
    [
        v2.ToImage(),
        v2.Resize(size=(384 - 24, 640)),
        v2.Pad(padding=[0, 12, 0, 12], fill=0),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

decoder = BBoxDecoder(CONF, NUM_BBOXES, REG_MAX, device)


# ---------------------------------------------------------
# 4. 실시간 메인 추론 엔진
# ---------------------------------------------------------
def main():
    mp.set_start_method("spawn", force=True)
    stop_signal = mp.Event()

    # ⭐️ [수정 2] LocklessBuffer 생성 (메인 프로세스가 Owner가 되어 메모리 할당 관리)
    buffer = LocklessBuffer(name=SHM_NAME, is_owner=True)

    # ⭐️ [수정 3] lock 인자 제거하고 runner 호출
    p = mp.Process(target=runner, args=(SHM_NAME, stop_signal))
    p.start()

    model = DetectionModel(num_classes=NUM_CLASSES).to(device)
    print(f"\n📦 모델 가중치({MODEL_PATH}) 로드 중...")
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)
        print("-> [NMS-Free 모델 가중치 로드 성공]")
    else:
        print(f"❌ [에러] 가중치 파일({MODEL_PATH})이 존재하지 않습니다.")
        stop_signal.set()
        p.join()
        buffer.close()
        return 1
    model.eval()

    dummy_input = torch.zeros(1, 3, 384, 640, device=device)
    for _ in range(3):
        _ = model(dummy_input)

    prev_ts = 0.0
    prev_time = time.time()
    fps = 0.0
    try:
        while True:
            # ⭐️ [수정 4] Lock 없이 최신 프레임을 즉시 Zero-copy 구조로 읽기
            current_frame = buffer.read_latest_frame()

            # 동일한 타임스탬프 프레임이면 추론 스킵 (Polling 대기)
            if current_frame.timestamp == prev_ts or current_frame.timestamp == 0.0:
                time.sleep(0.002)
                continue

            prev_ts = current_frame.timestamp

            color = current_frame.color
            depth = current_frame.depth
            view = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

            input_tensor = transform(color)[None, ...]
            input_tensor = input_tensor.to(device)

            with torch.inference_mode():
                cls_pred, reg_pred = model(input_tensor)

                decoded_bboxes = decoder.decode(reg_pred)[0]
                pred_scores = cls_pred[0].sigmoid().squeeze(-1)

                keep = pred_scores > CONF_THRES
                final_boxes = decoded_bboxes[keep]
                final_scores = pred_scores[keep]

            measured_3d = None
            if len(final_boxes) > 0:
                final_boxes = final_boxes.clone()
                final_boxes[:, [0, 2]] = final_boxes[:, [0, 2]] * 2
                final_boxes[:, [1, 3]] = (final_boxes[:, [1, 3]] - 12) * 2

                best_idx = torch.argmax(final_scores)
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

                # 3D Pos & Speed 텍스트
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

            if key == ord("q"):
                break

    finally:
        cv2.destroyAllWindows()
        stop_signal.set()
        p.join(timeout=3)
        if p.is_alive():
            p.terminate()
        buffer.close()
        print("[메인] 시스템이 안전하게 종료되었습니다.")


if __name__ == "__main__":
    main()

    # pipeline = None
    # try:
    #     # [Step 1] 왜곡 보정 맵 생성
    #     mapx, mapy = init_undistort_maps(
    #         CALIBRATION_JSON_PATH, COLOR_WIDTH, COLOR_HEIGHT
    #     )

    #     # [Step 2] 카메라 초기화
    #     print("📷 Orbbec Gemini 336 카메라 가동 중...")
    #     cam_device = get_first_device()
    #     load_depth_preset(cam_device)
    #     apply_color_exposure_settings(cam_device)

    #     pipeline = Pipeline()
    #     config = build_config(pipeline)
    #     pipeline.enable_frame_sync()
    #     pipeline.start(config)
    #     print("-> [카메라 파이프라인 가동 완료]")

    #     # [Step 3] NMS-Free 모델 불러오기

    #     # 워밍업 (Warm-up)

    #     window_name = "Gemini 336 RealTime Chamae NMS-Free Detector"
    #     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    #     cv2.resizeWindow(window_name, COLOR_WIDTH, COLOR_HEIGHT)
    #     print("\n🚀 실시간 추론을 시작합니다. (종료: 'q' 또는 ESC)")

    #     prev_time = time.time()
    #     fps = 0.0

    #     while True:
    #         # 1) 동기화 프레임 획득
    #         frames = pipeline.wait_for_frames(1500)
    #         if frames is None:
    #             if cv2.waitKey(1) & 0xFF in (ord("q"), ESC_KEY):
    #                 break
    #             continue

    #         color_frame = frames.get_color_frame()
    #         if color_frame is None:
    #             if cv2.waitKey(1) & 0xFF in (ord("q"), ESC_KEY):
    #                 break
    #             continue

    #         # RAW BGR 배열 추출 및 실시간 왜곡 보정
    #         raw_bgr_img = color_frame_to_bgr(color_frame)
    #         # if mapx is not None and mapy is not None:
    #         # bgr_img = cv2.remap(raw_bgr_img, mapx, mapy, cv2.INTER_LINEAR)
    #         # else:
    #         bgr_img = raw_bgr_img

    #         # orig_h, orig_w = bgr_img.shape[:2]

    #         # 2) 전처리 (Letterbox -> 384x640 텐서)
    #         # input_tensor, (r, dw, dh) = letterbox_preprocessing(
    #         # bgr_img, new_shape=MODEL_IMG_SIZE
    #         # )
    #         # 5) 원본 해상도로 좌표 역산 (GPU GPU-Accelerated 연산)
    #         display_img = bgr_img.copy()
    #         # if len(final_boxes) > 0:
    #         #     # 🛠️ [보정 2] GPU 텐서 상태에서 좌표 역산 및 경계선 락인 수행
    #         #     final_boxes[:, [0, 2]] = (final_boxes[:, [0, 2]] - dw) / r
    #         #     final_boxes[:, [1, 3]] = (final_boxes[:, [1, 3]] - dh) / r
    #         #     final_boxes[:, [0, 2]] = final_boxes[:, [0, 2]].clamp(0, orig_w)
    #         #     final_boxes[:, [1, 3]] = final_boxes[:, [1, 3]].clamp(0, orig_h)

    #         #     final_boxes = final_boxes.cpu().numpy()
    #         #     final_scores = final_scores.cpu().numpy()

    #         #     # 박스 드로잉
    #         # 🛠️ [보정 3] 실시간 FPS 계산 및 오버레이 표기
    #         curr_time = time.time()
    #         fps = 0.9 * fps + 0.1 * (1.0 / (curr_time - prev_time + 1e-6))
    #         prev_time = curr_time

    #         cv2.putText(
    #             display_img,
    #             f"FPS: {fps:.1f}",
    #             (15, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             0.8,
    #             (0, 0, 255),
    #             2,
    #             cv2.LINE_AA,
    #         )

    #         # 6) 화면 출력
    #         cv2.imshow(window_name, display_img)
    #         key = cv2.waitKey(1) & 0xFF
    #         if key in (ord("q"), ESC_KEY):
    #             break

    #     cv2.destroyAllWindows()
    #     return 0
    # except Exception as exc:
    #     print(f"❌ [에러 발생]: {exc}")
    #     import traceback

    #     traceback.print_exc()
    #     return 1
    # finally:
    #     if pipeline is not None:
    #         try:
    #             pipeline.stop()
    #             print("[INFO] 카메라 파이프라인 안전 종료 완료.")
    #         except Exception:
    #             pass


if __name__ == "__main__":
    raise SystemExit(main())
