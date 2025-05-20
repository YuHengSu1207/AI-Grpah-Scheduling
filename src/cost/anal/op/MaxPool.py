from . import *

def analysis(model:onnx.ModelProto,layout:str, node:onnx.NodeProto, memoryTable:Optional[list] = None,
        csvPath:Optional[str] = None, pervious_cycle:Optional[int] = 0, use_count: dict = None) -> tuple([int, dict,list]):
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
    
    # Get kernel_shape, stride, pad if available
    kernel = [1, 1]
    stride = [1, 1]
    pads = [0, 0, 0, 0]

    for attr in node.attribute:
        if attr.name == "kernel_shape":
            kernel = list(attr.ints)
        elif attr.name == "strides":
            stride = list(attr.ints)
        elif attr.name == "pads":
            pads = list(attr.ints)

    kernel_h, kernel_w = kernel
    stride_h, stride_w = stride
    pad_top, pad_left, pad_bottom, pad_right = pads if len(pads) == 4 else (0, 0, 0, 0)

    N, C, H_out, W_out = dimY  # Output shape

    # ----------- MaxPool Operation Estimation -----------
    # For each output element: kernel_h * kernel_w reads and compares
    comparisons_per_output = kernel_h * kernel_w
    total_outputs = N * C * H_out * W_out
    cycle = total_outputs * comparisons_per_output  # Assume each comparison roughly 1 cycle

    memory = staticMemX + staticMemY

    # ----------- Memory Management -----------
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