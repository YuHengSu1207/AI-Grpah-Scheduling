from . import *

def analysis(model:onnx.ModelProto,layout:str, node:onnx.NodeProto, memoryTable:Optional[list] = None, csvPath:Optional[str] = None, pervious_cycle:Optional[int] = 0) -> tuple([int, dict, list]):
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

    for ipt in node.input:
        memoryTable = tool.free(ipt, memoryTable)

    return memoryRequest, {"memory" : memory / 8192, "cycle":int(cycle)}, memoryTable