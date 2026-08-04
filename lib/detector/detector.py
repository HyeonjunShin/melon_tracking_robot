from model import DetectionModel
import openvino as ov
import os

def preprocess(input):
    pass
    

class Detector:
    def __init__(self):
        self.model = DetectionModel()
        self.conf_threshold = 0.8
        self.model_path = "model_int8.xml"

        core = ov.Core()
        device_name = "GPU" if "GPU" in core.available_devices else "CPU"
        print(f"📦 OpenVINO INT8 모델 로드 중... [디바이스: {device_name}]")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(1, "Not found the model file.", self.model_path)
        
        ov_model = core.read_model(self.model_path)
        self.compiled_model = core.compile_model(ov_model, device_name)

        # Binding the output layer.
        self.output_0 = self.compiled_model.output(0)
        self.output_1 = self.compiled_model.output(1)

    def detect(self, input):
        

        results = self.compiled_model({0: input_tensor})

        