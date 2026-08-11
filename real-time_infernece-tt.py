import json
import multiprocessing as mp
import os
import time
import cv2
import numpy as np
import openvino as ov

from lib.camera.gemini336 import CameraBuffer, camera_runner

# from lib.tracker.tracker import KalmanFilter3D, draw_3d_tf_axis

ESC_KEY = 27

FX, FY = 693.3102, 693.4061
CX, CY = 639.6599, 365.0724

# kf_tracker = KalmanFilter3D(dt=1 / 30.0)
INT8_MODEL_PATH = "./model_int8.xml"


def main():
    mp.set_start_method("spawn", force=True)
    stop_signal = mp.Event()

    camera_buffer = CameraBuffer(shm_name="test", is_owner=True)
    camera_process = mp.Process(
        target=camera_runner, args=("test", stop_signal)
    )
    camera_process.start()

    core = ov.Core()
    device_name = "GPU" if "GPU" in core.available_devices else "CPU"
    print(f"📦 OpenVINO INT8 모델 로드 중... [디바이스: {device_name}]")

    if not os.path.exists(INT8_MODEL_PATH):
        print(f"❌ [에러] 모델 파일({INT8_MODEL_PATH})이 존재하지 않습니다.")
        stop_signal.set()
        camera_process.join()
        camera_buffer.close()
        return 1

    ov_model = core.read_model(INT8_MODEL_PATH)
    compiled_model = core.compile_model(ov_model, device_name)

    # Binding the output layer.
    output_0 = compiled_model.output(0)
    output_1 = compiled_model.output(1)

    prev_ts = 0.0
    prev_time = time.time()
    fps = 0.0

    print("🚀 동기식 추론 및 3D 추적 루프가 시작되었습니다.")
    is_recording = False
    video_writer = None
    output_filename = "output_record.mp4"

    try:
        while True:
            if not camera_buffer.get_status():
                continue

            current_frame = camera_buffer.read_latest_frame()
            if current_frame.timestamp == prev_ts:
                time.sleep(0.001)
                continue

            prev_ts = current_frame.timestamp
            color = current_frame.color
            depth = current_frame.depth
            view = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

            input_tensor = color[None, ...]
            results = compiled_model({0: input_tensor})

            scores = results[output_0].squeeze()
            bboxes = results[output_1][0]

            best_idx = np.argmax(scores)

            # keep = scores > 0.8
            final_boxes = bboxes[best_idx]
            final_scores = scores[best_idx]

            final_boxes[[0, 2]] = final_boxes[[0, 2]] * 2
            final_boxes[[1, 3]] = (final_boxes[[1, 3]] - 12) * 2
            x1, y1, x2, y2 = map(int, final_boxes)
            cv2.rectangle(view, (x1, y1), (x2, y2), (255, 0, 0), 3)

            # measured_3d = None
            # if len(final_boxes) > 0 and len(final_scores) > 0:
            #     final_boxes[:, [0, 2]] = final_boxes[:, [0, 2]] * 2
            #     final_boxes[:, [1, 3]] = (final_boxes[:, [1, 3]] - 12) * 2

            #     best_idx = np.argmax(final_scores)
            #     best_box = final_boxes[best_idx].tolist()

            #     # Depth 기반 3D Centroid 측정
            #     # measured_3d = kf_tracker.compute_3d_centroid(
            #     # depth, best_box, FX, FY, CX, CY
            #     # )

            #     x1, y1, x2, y2 = map(int, best_box)
            #     cv2.rectangle(view, (x1, y1), (x2, y2), (255,0,0), 3)

            # 5. 3D Kalman Filter 업데이트
            # state, is_updated = kf_tracker.update(measured_3d)

            # 6. 제어 프로세스로 추적 결과 전송 (UDP)
            # if kf_tracker.is_initialized:
            # xc, yc, zc, vx, vy, vz = state

            # control_packet = {
            #     "timestamp": float(current_frame.timestamp),
            #     "pos_mm": [float(xc), float(yc), float(zc)],
            #     "vel_mms": [float(vx), float(vy), float(vz)],
            #     "is_valid": bool(is_updated),
            # }
            # # 3D TF 축 시각화
            # draw_3d_tf_axis(
            #     img=view,
            #     center_3d=(xc, yc, zc),
            #     fx=FX,
            #     fy=FY,
            #     cx=CX,
            #     cy=CY,
            #     axis_length=80.0,
            #     thickness=2,
            # )

            # 화면 텍스트 출력
            # pos_text = f"TF [X:{xc:.0f}, Y:{yc:.0f}, Z:{zc:.0f}] mm"
            # vel_text = f"V [Vx:{vx:.0f}, Vy:{vy:.0f}, Vz:{vz:.0f}] mm/s"
            # cv2.putText(
            #     view,
            #     pos_text,
            #     (10, 30),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.6,
            #     (0, 255, 255),
            #     2,
            #     cv2.LINE_AA,
            # )
            # cv2.putText(
            #     view,
            #     vel_text,
            #     (10, 60),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.6,
            #     (255, 255, 0),
            #     2,
            #     cv2.LINE_AA,
            # )

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

            if is_recording and video_writer is not None:
                video_writer.write(view)
                cv2.putText(
                    view,
                    "● REC",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("color", view)

            key = cv2.waitKey(1) & 0xFF
            if key == 32:
                if not is_recording:
                    # 녹화 시작 세팅
                    height, width, _ = view.shape
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    # FPS는 실제 루프 도는 속도에 맞춰 조절 필요 (예: 30 FPS)
                    video_writer = cv2.VideoWriter(
                        output_filename, fourcc, 30.0, (width, height)
                    )
                    is_recording = True
                    print("▶️ 녹화를 시작합니다.")
                else:
                    # 녹화 중지 및 파일 닫기 (자동 저장)
                    is_recording = False
                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None
                    print(
                        f"⏹️ 녹화를 종료하고 저장했습니다. 파일명: {output_filename}"
                    )

            # 2. ESC 키(ASCII 27) 또는 'q'를 누르면 전체 루프 종료
            elif key == 27 or key == ord("q"):
                print("프로그램을 종료합니다.")
                break

    finally:
        cv2.destroyAllWindows()
        stop_signal.set()
        camera_process.join(timeout=3)
        if camera_process.is_alive():
            camera_process.terminate()
        camera_buffer.close()
        print("[메인] 시스템이 안전하게 종료되었습니다.")


if __name__ == "__main__":
    raise SystemExit(main())
