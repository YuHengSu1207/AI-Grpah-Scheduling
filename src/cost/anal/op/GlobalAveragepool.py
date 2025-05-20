from . import *

def analysis(model:onnx.ModelProto,layout:str, node:onnx.NodeProto, memoryTable:Optional[list] = None, 
        csvPath:Optional[str] = None, pervious_cycle:Optional[int] = 0, use_count: dict = None) -> tuple([int, dict, list]):
    memory = 0
    cycle = 0
    memoryRequest = 0

    _, X = get_value_info(node.input[0], model)
    _, Y = get_value_info(node.output[0], model)

    dimX = [d.dim_value for d in X.type.tensor_type.shape.dim]
    dimY = [d.dim_value for d in Y.type.tensor_type.shape.dim]

    if layout == "NHWC":
        dimX[1], dimX[3] = dimX[3], dimX[1]
        dimY[1], dimY[3] = dimY[3], dimY[1]

    channels = dimX[1]
    H, W = dimX[2], dimX[3]
    pool_area = H * W

    typeX = X.type.tensor_type.elem_type
    typeY = Y.type.tensor_type.elem_type

    staticMemX = DATA_SIZE_DTYPE[typeX]
    staticMemY = DATA_SIZE_DTYPE[typeY]
    for dim in dimX:
        staticMemX *= dim
    for dim in dimY:
        staticMemY *= dim

    num_output_elements = 1
    for dim in dimY:
        num_output_elements *= dim

    # Each output element requires reading N pixels, summing, dividing, storing
    cost_per_output = pool_area + 1 + 1  # sum + div + store
    cycle = num_output_elements * cost_per_output

    memory = staticMemX + staticMemY

    request, memoryTable = tool.malloc(node.output[0], staticMemY // 8, memoryTable)
    memoryRequest += request
    _ = tool.dump_csv(csvPath=csvPath, memoryTable=memoryTable, memMAX=config.MEMORY_SIZE, second=cycle + pervious_cycle)

    # After using inputs
    for input_name in node.input:
        if use_count is not None:
            use_count[input_name] -= 1
            if use_count[input_name] == 0:
                memoryTable = tool.free(input_name, memoryTable)
                
    return memoryRequest, {"memory" : memory / 8192, "cycle":int(cycle)}, memoryTable, use_count