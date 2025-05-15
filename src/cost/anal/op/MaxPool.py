from . import *

def analysis(model:onnx.ModelProto,layout:str, node:onnx.NodeProto, memoryTable:Optional[list] = None, csvPath:Optional[str] = None) -> tuple([int, dict,list]):
    memory = 0
    cycle = 0
    memoryRequest = 0
    
    _, X = get_value_info(node.input[0], model)
    _, Y = get_value_info(node.output[0], model)
    dimX = [a.dim_value for a in X.type.tensor_type.shape.dim]
    dimY = [a.dim_value for a in Y.type.tensor_type.shape.dim]
    if layout == "NHWC":
        dimX[1], dimX[3] = dimX[3], dimX[1]
        dimY[1], dimY[3] = dimY[3], dimY[1] 
    typeX = X.type.tensor_type.elem_type    
    typeY = Y.type.tensor_type.elem_type
    
    staticMemX = DATA_SIZE_DTYPE[typeX]
    staticMemY = DATA_SIZE_DTYPE[typeY]
    
    for dim in dimX: staticMemX *= dim
    for dim in dimY: staticMemY *= dim

    ######### memory Management #########
    request, memoryTable = tool.malloc(node.output[0], staticMemY // 8 , memoryTable)
    memoryRequest += request
    _ = tool.dump_csv(csvPath=csvPath, memoryTable=memoryTable, memMAX=config.MEMORY_SIZE_IN_LAB16_3+1, second=cycle)
    # for ipt in node.input:
    #     memoryTable = tool.free(ipt, memoryTable)
    memoryTable = tool.free(node.input[0], memoryTable)
    ######### memory Management #########
    return memoryRequest, {"memory" : memory / 8192, "cycle":int(cycle)}, memoryTable