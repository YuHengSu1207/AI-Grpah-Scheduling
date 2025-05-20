from . import *

def analysis(model:onnx.ModelProto,layout:str, node:onnx.NodeProto, memoryTable:Optional[list] = None, 
            csvPath:Optional[str] = None, pervious_cycle:Optional[int] = 0, use_count: dict = None) -> tuple([int, dict,list]):
    memory = 0
    cycle = 0
    memoryRequest = 0
    
    # Input = Output size
    # Each output element is computed as max(0, x)
    # For scalar CPU: this is typically 1 load + 1 compare + 1 store per element.
    
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

    # ----------- ReLU Operation Estimation -----------
    num_elements = 1
    for dim in dimY:
        num_elements *= dim

    # Each element: load input, compare with zero, write output
    # Cost: 1 load + 1 cmp + 1 store roughly 3 cycles per element
    relu_cost_per_element = 3  # scalar CPU
    cycle = num_elements * relu_cost_per_element

    memory = staticMemX + staticMemY
    
    ######### memory Management #########
    request, memoryTable = tool.malloc(node.output[0], staticMemY // 8 , memoryTable)
    memoryRequest += request
    _ = tool.dump_csv(csvPath=csvPath, memoryTable=memoryTable, memMAX=config.MEMORY_SIZE, second=cycle + pervious_cycle)
    
    # After using inputs
    for input_name in node.input:
        if use_count is not None:
            use_count[input_name] -= 1
            if use_count[input_name] == 0:
                memoryTable = tool.free(input_name, memoryTable)
    ######### memory Management #########
    return memoryRequest, {"memory" : memory / 8192, "cycle":int(cycle)}, memoryTable, use_count