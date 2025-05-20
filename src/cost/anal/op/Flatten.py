from . import *

def analysis(model:onnx.ModelProto,layout:str, node:onnx.NodeProto, memoryTable:Optional[list] = None,
        csvPath:Optional[str] = None, pervious_cycle:Optional[int] = 0, use_count: dict = None) -> tuple([int, dict, list]):
    memory = 0
    cycle = 0
    memoryRequest = 0

    _, X = get_value_info(node.input[0], model)
    _, Y = get_value_info(node.output[0], model)

    dimX = [d.dim_value for d in X.type.tensor_type.shape.dim]
    typeX = X.type.tensor_type.elem_type
    staticMemX = DATA_SIZE_DTYPE[typeX]
    for dim in dimX:
        staticMemX *= dim

    # Flatten is just a memory view change. Assume 1 cycle per element to move.
    num_elements = 1
    for dim in dimX:
        num_elements *= dim

    cycle = num_elements  # 1 cycle per element
    memory = staticMemX * 2  # read and write

    request, memoryTable = tool.malloc(node.output[0], staticMemX // 8, memoryTable)
    memoryRequest += request
    _ = tool.dump_csv(csvPath=csvPath, memoryTable=memoryTable, memMAX=config.MEMORY_SIZE, second=cycle + pervious_cycle)

    # After using inputs
    for input_name in node.input:
        if use_count is not None:
            use_count[input_name] -= 1
            if use_count[input_name] == 0:
                memoryTable = tool.free(input_name, memoryTable)
    
    return memoryRequest, {"memory" : memory / 8192, "cycle":int(cycle)}, memoryTable