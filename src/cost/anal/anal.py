import onnx
from .op import Conv, Relu, template, MaxPool
from typing import Optional, Union
def analyticalModel(model:onnx.ModelProto,layout:str, node:onnx.NodeProto, memoryTable:list = None, csvPath:str = None, previous_cycle: int = 0) -> tuple([int, dict,list]):
    operator = {
        "Conv" : Conv.analysis,
        "Relu" : Relu.analysis,
        "MaxPool": MaxPool.analysis,
        "Add" :  template.analysis,
        "GlobalAveragePool" : template.analysis,
        "Flatten" : template.analysis,
        "Gemm" : template.analysis,
        "Concat" : template.analysis,
        "Sum" : template.analysis
        }
    if node.op_type not in operator.keys():
        raise BaseException(f"Analytical Model : \'{node.op_type}\' doesn't exist." )
    else:
        return operator[node.op_type](model, layout, node, memoryTable, csvPath, previous_cycle)

    
