import onnx
from onnx import shape_inference



def dims(shape_proto):
    return [d.dim_value or d.dim_param or "?" for d in shape_proto.dim]

def dtype_name(elem_type):
    return onnx.TensorProto.DataType.Name(elem_type).lower() if elem_type else None

def load_model(name):
    model = shape_inference.infer_shapes(onnx.load("yolov8n.onnx"))
    g = model.graph
    return g

def load_tensor_info(g):
    dim_info = {}
    for vi in [*g.input, *g.value_info, *g.output]:
        tt = vi.type.tensor_type
        dim_info[vi.name] = (dims(tt.shape), dtype_name(tt.elem_type))
    weigths={}
    for vi in g.initializer:
        weigths[vi.name]=(list(vi.dims),dtype_name(tt.elem_type))
    return  dim_info,weigths

graph=load_model("yolov8n.onnx")
print(load_tensor_info(graph)[1])