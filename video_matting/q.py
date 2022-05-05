import onnx
from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType

model = onnx.load("rvm_mobilenetv3_fp32.onnx")
nodes_to_exclude = []
# nodes_to_exclude += [x.name for x in model.graph.node if x.name.startswith('Conv')]

quantized_model = quantize_dynamic(
    'rvm_mobilenetv3_fp32.onnx',
    'rvm_mobilenetv3_int8.onnx',
    nodes_to_exclude=nodes_to_exclude,
    weight_type=QuantType.QUInt8
)
