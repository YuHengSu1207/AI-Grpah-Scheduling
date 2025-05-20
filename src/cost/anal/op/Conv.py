from . import *

def analysis(model:onnx.ModelProto,layout:str, node:onnx.NodeProto, memoryTable:Optional[list] = None, 
            csvPath:Optional[str] = None, pervious_cycle:Optional[int] = 0, use_count: dict = None) -> tuple([int, dict,list]):
    memoryRequest = 0
    _, X = get_value_info(node.input[0], model)
    
    _, W = get_initilizer(node.input[1], model)
    _, Y = get_value_info(node.output[0], model)
    
    if len(node.input) == 3:
        _, B = get_initilizer(node.input[2], model)
        # print(B.dims[0])
    
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

    # Extract stride, padding, dilation from attributes
    stride = [1, 1]
    padding = [0, 0, 0, 0]  # [pad_top, pad_left, pad_bottom, pad_right]
    dilation = [1, 1]

    for attr in node.attribute:
        if attr.name == "strides":
            stride = list(attr.ints)
        elif attr.name == "pads":
            padding = list(attr.ints)
        elif attr.name == "dilations":
            dilation = list(attr.ints)

    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation
    pad_top, pad_left, pad_bottom, pad_right = padding
    
    # Kernel size
    k_cout, k_cin, k_h, k_w = dimW

    # Input size
    n, c_in, h_in, w_in = dimX
    n_out, c_out, h_out, w_out = dimY

    # Total output elements = store count
    store = n_out * c_out * h_out * w_out
    
    # LW instruction :
    # Recalculate the number of input loads
    load = 0
    for h in range(h_out):
        for w in range(w_out):
            for kh in range(k_h):
                for kw in range(k_w):
                    h_in_idx = h * stride_h - pad_top + kh * dilation_h
                    w_in_idx = w * stride_w - pad_left + kw * dilation_w
                    if 0 <= h_in_idx < h_in and 0 <= w_in_idx < w_in:
                        load += n * c_in * c_out  # one load per input-channel × batch × output-channel
                        
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
    create_output = (config.DATA_LATENCY + create_output_address) * store + create_temp * store * k_cin * k_h * k_w
    # Number of branch
    branch_count = 0
    # CPU instruction count :
    cycle = create_input + create_kernel + create_output + branch_count
    # Memory Requirement
    memory = staticMemX + staticMemY + staticMemW
    
    ######### memory Management #########
    request, memoryTable = tool.malloc(node.output[0], staticMemY // 8 , memoryTable)
    memoryRequest += request
    tool.dump_csv(csvPath=csvPath, memoryTable=memoryTable, memMAX=config.MEMORY_SIZE, second=cycle + pervious_cycle)

    # After using inputs
    for input_name in node.input:
        if use_count is not None:
            use_count[input_name] -= 1
            if use_count[input_name] == 0:
                memoryTable = tool.free(input_name, memoryTable)
        
    ######### memory Management #########
    return memoryRequest, {"memory" : memory / 8192, "cycle":int(cycle)}, memoryTable, use_count
    

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
    
    
