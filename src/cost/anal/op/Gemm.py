from . import *

def analysis(model:onnx.ModelProto,layout:str, node:onnx.NodeProto, memoryTable:Optional[list] = None, csvPath:Optional[str] = None, pervious_cycle:Optional[int] = 0) -> tuple([int, dict, list]):
    memory = 0
    cycle = 0
    memoryRequest = 0

    _, A = get_value_info(node.input[0], model)
    _, B = get_initilizer(node.input[1], model)
    _, C = get_value_info(node.output[0], model)

    dimA = [d.dim_value for d in A.type.tensor_type.shape.dim]
    dimB = B.dims
    dimC = [d.dim_value for d in C.type.tensor_type.shape.dim]

    print(dimA)
    print(dimB)
    print(dimC)
    M = dimA[0]
    K = dimA[1]
    assert(dimA[1] == dimB[1])
    N = dimB[0]

    typeA = A.type.tensor_type.elem_type
    typeB = B.data_type
    typeC = C.type.tensor_type.elem_type

    staticMemA = DATA_SIZE_DTYPE[typeA]
    staticMemB = DATA_SIZE_DTYPE[typeB]
    staticMemC = DATA_SIZE_DTYPE[typeC]

    for dim in dimA:
        staticMemA *= dim
    for dim in dimB:
        staticMemB *= dim
    for dim in dimC:
        staticMemC *= dim

    # GEMM: M×N output, each is dot of K elements
    # Each output: K loads (A) + K loads (B) + K mul + K-1 adds + 1 store
    ops_per_output = K * 2 + K + (K - 1) + 1
    cycle = M * N * ops_per_output

    memory = staticMemA + staticMemB + staticMemC

    request, memoryTable = tool.malloc(node.output[0], staticMemC // 8, memoryTable)
    memoryRequest += request
    _ = tool.dump_csv(csvPath=csvPath, memoryTable=memoryTable, memMAX=config.MEMORY_SIZE, second=cycle + pervious_cycle)

    for ipt in node.input:
        memoryTable = tool.free(ipt, memoryTable)
        
    return memoryRequest, {"memory" : memory / 8192, "cycle":int(cycle)}, memoryTable