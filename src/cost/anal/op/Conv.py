from . import *

def analysis(model:onnx.ModelProto,layout:str, node:onnx.NodeProto, memoryTable:Optional[list] = None, csvPath:Optional[str] = None) -> tuple([int, dict,list]):
    return analysis_scalar_memory_dependent(model,layout,node, memoryTable, csvPath)
    # return analysis_matrix_memory_reallocated(model,layout,node)

def analysis_scalar_memory_dependent(model:onnx.ModelProto,layout:str, node:onnx.NodeProto, memoryTable:Optional[list] = None, csvPath:Optional[str] = None) -> tuple([int, dict,list]):
    memoryRequest = 0
    _, X = get_value_info(node.input[0], model)
    
    _, W = get_initilizer(node.input[1], model)
    _, Y = get_value_info(node.output[0], model)
    
    if len(node.input) == 3:
        _, B = get_initilizer(node.input[2], model)
    
    dimX = [a.dim_value for a in X.type.tensor_type.shape.dim]
    dimY = [a.dim_value for a in Y.type.tensor_type.shape.dim]
    dimW = W.dims
    if layout == "NHWC":
        dimX[1], dimX[3] = dimX[3], dimX[1]
        dimW[1], dimW[3] = dimW[3], dimW[1]
        dimY[1], dimY[3] = dimY[3], dimY[1] 
    typeX = X.type.tensor_type.elem_type    
    typeY = Y.type.tensor_type.elem_type
    typeW = W.data_type
    
    staticMemX = DATA_SIZE_DTYPE[typeX]
    staticMemY = DATA_SIZE_DTYPE[typeY]
    staticMemW = DATA_SIZE_DTYPE[typeW]
    
    for dim in dimX: staticMemX *= dim
    for dim in dimY: staticMemY *= dim
    for dim in dimW: staticMemW *= dim

    loop = dimY[0] * dimY[1] * dimY[2] * dimY[3] * dimW[1] * dimW[2] * dimW[3] 
    
    # SW instruction :
    store = dimY[0] * dimY[1] * dimY[2] * dimY[3]
    # LW instruction :
    load = loop
    # Create Address for input kernel output
    create_input_pivot = 3 + 2 # 3 (multiply) + 2 (addition) 
    create_kernel_address =  3 + 3 # 3 (multiply) + 3 (addition)
    create_input_address =  2 + 3 + (create_input_pivot / dimW[1] * dimW[2] * dimW[3])# 2 (multiply) + 3 (addition)
    create_output_address = 2 + 3 # 2 (multiply) + 3 (addition)
    create_temp = 2 + 3 # 2 (multiply) + 3 (addition)
    # create Input
    create_input = (config.DATA_LATENCY + create_input_address) * load
    # Create Kernel
    create_kernel = (config.DATA_LATENCY + create_kernel_address) * load
    # Create Output
    create_output = (config.DATA_LATENCY + create_output_address) * store + create_temp * loop
    # Number of branch
    branch_count = 0
    # CPU instruction count :
    cycle = create_input + create_kernel + create_output + branch_count
    # Memory Requirement
    memory = staticMemX + staticMemY + staticMemW
    

    ######### memory Management #########
    request, memoryTable = tool.malloc(node.output[0], staticMemY // 8 , memoryTable)
    memoryRequest += request
    tool.dump_csv(csvPath=csvPath, memoryTable=memoryTable, memMAX=config.MEMORY_SIZE_IN_LAB16_3+1, second=cycle)
    # for ipt in node.input:
    #     memoryTable = tool.free(ipt, memoryTable)
    memoryTable = tool.free(node.input[0], memoryTable)
    ######### memory Management #########
    return memoryRequest, {"memory" : memory / 8192, "cycle":int(cycle)}, memoryTable
    

def analysis_matrix_memory_reallocated(model:onnx.ModelProto,layout:str, node:onnx.NodeProto, memoryTable:Optional[list] = None, csvPath:Optional[str] = None) -> tuple([int, dict,list]):
    
    busBandwidth = 32
    
    _, X = get_value_info(node.input[0], model)
    _, W = get_initilizer(node.input[1], model)
    _, Y = get_value_info(node.output[0], model)
    
    if len(node.input) == 3:
        _, B = get_value_info(node.input[2], model.graph)
    
    dimX = [a.dim_value for a in X.type.tensor_type.shape.dim]
    dimY = [a.dim_value for a in Y.type.tensor_type.shape.dim]
    dimW = W.dims
    
    if layout == "NCHW":
        dimX[1], dimX[3] = dimX[3], dimX[1]
        dimW[1], dimW[3] = dimW[3], dimW[1]
        dimY[1], dimY[3] = dimY[3], dimY[1]         
    elif layout == "NHWC":
        dimX, dimY, dimW = dimX, dimY, dimW

    typeX = X.type.tensor_type.elem_type    
    typeY = Y.type.tensor_type.elem_type
    typeW = W.data_type
    
    dilations, group, kernel_shape, pads, strides = get_attribute(model.graph.node[0].attribute)
    
    
    staticMemX = DATA_SIZE_DTYPE[typeX]
    staticMemY = DATA_SIZE_DTYPE[typeY]
    staticMemW = DATA_SIZE_DTYPE[typeW]
    
    ##################################
    #              im2col            #
    ##################################
    # ----------- Output ----------- #
    for dim in dimY: staticMemY *= dim
    # ----------- Kernel ----------- #
    for dim in dimW: staticMemW *= dim
    staticMemW_im2col = 0
    loadW_cycle = 0
    storeW_cycle = 0
    if layout == "NHWC":
        staticMemW_im2col = staticMemW
        loadW_cycle = staticMemW / busBandwidth
        storeW_cycle = staticMemW / DATA_SIZE_DTYPE[typeW]
    # ----------- Input  ----------- #
    for dim in dimX: staticMemX *= dim
    staticMemX_im2col = 0
    loadX_cycle = 0
    storeX_cycle = 0
    if layout == "NHWC":
        staticMemX_im2col = staticMemX
        loadX_cycle = staticMemX / busBandwidth
        storeX_cycle = staticMemX / DATA_SIZE_DTYPE[typeX]
    # -----------  Bias  ----------- #
    staticMemB = 0
    if len(model.graph.initializer) > 1:
        B = model.graph.initializer[1]
        dimB = B.dims
        typeB = B.data_type
        staticMemB = DATA_SIZE_DTYPE[typeB]
        for dim in dimB: staticMemB *= dim
    
    wi = staticMemW + staticMemX
    wWi = staticMemW + staticMemW_im2col + staticMemX
    WiI = staticMemW_im2col + staticMemX + staticMemX_im2col
    WI = staticMemW_im2col + staticMemX_im2col
    WIo = staticMemW_im2col + staticMemX_im2col + staticMemY
    
    memory = max(wi, wWi, WiI, WI, WIo) + staticMemB
    
    ##################################
    #      matrix multiplication     #
    ##################################
    # -----------  Compute  --------- => 16 MAC
    M = dimY[0] * dimY[1]
    N = dimW[1] * dimW[2] * dimW[3]
    K = dimY[3]

    dM = math.ceil(M/16)
    dN = math.ceil(N/16)
    dK = math.ceil(K/16)

    matmul_num = dM * dN * dK
    
    matmul_cycle = 256 * matmul_num + 105 + 7 + (staticMemB // 8)

    cycle = matmul_cycle + loadW_cycle + loadX_cycle + storeW_cycle + storeX_cycle

    return 0, {"memory" : memory / 8192, "cycle" : cycle}, memoryTable
    
    
