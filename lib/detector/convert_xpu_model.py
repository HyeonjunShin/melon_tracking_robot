import nncf
import openvino as ov
import torch
from torch.utils.data import DataLoader

from datasets import ChamaeDataset, chamae_collate_fn, get_data, split_data
from model import DualYOLOv8ChamaeModel

paired_path = get_data()
train_path, val_path = split_data(paired_path)

dataset = ChamaeDataset(train_path, False)
data_loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=chamae_collate_fn,
    num_workers=4,
)

calibration_tensors = []
for i, b in enumerate(data_loader):
    color_img, depth = b  # color_img shape: (1, 3, 384, 640)
    calibration_tensors.append(color_img)
    if len(calibration_tensors) >= 200:
        break

print(
    f"✅ Calibration 데이터 준비 완료: {len(calibration_tensors)}개 (Shape: {calibration_tensors[0].shape})"
)

model = DualYOLOv8ChamaeModel()
model.load_state_dict(torch.load("./lib/detector/weight.pt", map_location="cpu"))
model.eval()

dummy_input = torch.randn(1, 3, 384, 640)
ov_model = ov.convert_model(model, example_input=dummy_input)
ov.save_model(ov_model, "model_fp16.xml", compress_to_fp16=True)
print("1단계: FP16 변환 완료!")

nncf_dataset = nncf.Dataset(calibration_tensors)

print("🚀 INT8 NNCF Calibration 양자화 진행 중...")
quantized_model = nncf.quantize(ov_model, nncf_dataset)
ov.save_model(quantized_model, "model_int8.xml")

print("2단계: INT8 캘리브레이션 양자화 완료! (model_int8.xml)")
