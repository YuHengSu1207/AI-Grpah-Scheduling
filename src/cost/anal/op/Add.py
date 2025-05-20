from . import *

def analysis(model:onnx.ModelProto,layout:str, node:onnx.NodeProto, memoryTable:Optional[list] = None,
        csvPath:Optional[str] = None, pervious_cycle:Optional[int] = 0, use_count: dict = None) -> tuple([int, dict, list]):
    memory = 0
    cycle = 0
    memoryRequest = 0

    _, A = get_value_info(node.input[0], model)
    _, B = get_value_info(node.input[1], model)
    _, Y = get_value_info(node.output[0], model)

    dimY = [d.dim_value for d in Y.type.tensor_type.shape.dim]
    typeY = Y.type.tensor_type.elem_type

    staticMemY = DATA_SIZE_DTYPE[typeY]
    for dim in dimY:
        staticMemY *= dim

    # Each element: load A, load B, add, store
    num_elements = 1
    for dim in dimY:
        num_elements *= dim

    add_cost_per_element = 4  # 2 loads + 1 add + 1 store
    cycle = num_elements * add_cost_per_element

    memory = staticMemY * 3  # assume A, B, and output

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