import nncf
import openvino as ov
import torch
from torch.utils.data import DataLoader

from datasets import ChamaeDataset, chamae_collate_fn, get_data, split_data
from model import DetectionModel, CombinedModel
from openvino.preprocess import PrePostProcessor, ResizeAlgorithm, PaddingMode

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

base_model = DetectionModel(num_classes=1, reg_max=16)
base_model.load_state_dict(torch.load("./lib/detector/weight.pt", map_location="cpu"))

model = CombinedModel(base_model)
model.eval()

dummy_input = torch.randn(1, 3, 384, 640)
ov_model = ov.convert_model(model, example_input=dummy_input)
ov.save_model(ov_model, "model_fp16.xml", compress_to_fp16=True)
print("1단계: FP16 변환 완료!")

nncf_dataset = nncf.Dataset(calibration_tensors)

print("🚀 INT8 NNCF Calibration 양자화 진행 중...")
quantized_model = nncf.quantize(ov_model, nncf_dataset)


ppp = PrePostProcessor(model)
ppp.input().model().set_layout(ov.Layout("NCHW"))

ppp.input().tensor().set_shape([1, 720, 1280, 3]).set_element_type(
    ov.Type.u8
).set_layout(ov.Layout("NHWC"))

ppp.input().preprocess().resize(
    ResizeAlgorithm.RESIZE_LINEAR, 360, 640
).convert_element_type(ov.Type.f32).pad(
    pads_begin=[0, 12, 0, 0],
    pads_end=[0, 12, 0, 0],
    value=[0.0],
    mode=PaddingMode.CONSTANT,
).scale(
    255.0
)

model = ppp.build()

ov.save_model(quantized_model, "model_int8.xml")

print("2단계: INT8 캘리브레이션 양자화 완료! (model_int8.xml)")
