import onnx
from onnx import shape_inference

model=shape_inference.infer_shapes(onnx.load("yolov8n.onnx"))
g=model.graph

for n in g.node:
    print(n)