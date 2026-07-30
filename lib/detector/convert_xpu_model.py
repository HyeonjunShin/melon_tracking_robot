import torch
import openvino as ov
import nncf
from model import DualYOLOv8ChamaeModel

# 1. 학습된 모델 로드
model = DualYOLOv8ChamaeModel()
model.load_state_dict(torch.load("./lib/detector/weight.pt", map_location="cpu"))
model.eval()

# 2. PyTorch -> OpenVINO FP16 변환
dummy_input = torch.randn(1, 3, 384, 640) # 실제 모델 입력 크기에 맞게 수정
ov_model = ov.convert_model(model, example_input=dummy_input)
ov.save_model(ov_model, "model_fp16.xml", compress_to_fp16=True)

print("1단계: FP16 변환 완료! (크기 약 50% 축소)")

# 3. NNCF를 통한 INT8 수량화 (Calibration 데이터 준비)
# 실제 테스트 데이터 중 50~100장 정도의 이미지 Tensor 리스트 준비
calibration_samples = [torch.randn(1, 3, 384, 640) for _ in range(50)] 
dataset = nncf.Dataset(calibration_samples)

# INT8 변환
quantized_model = nncf.quantize(ov_model, dataset)
ov.save_model(quantized_model, "model_int8.xml")

print("2단계: INT8 변환 완료! (크기 약 75% 축소, Intel iGPU 최적화 완료)")