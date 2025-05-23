
import onnx 
from src.scheduler.scheduler import scheduler
from src.tool.util import create_operator_list_dict, create_tensor_dict
from src.management import tool
from src.structure import DATA_SIZE_DTYPE
import os
import src.config as config
import math
from collections import defaultdict

def manager(model:onnx.ModelProto, operatorPath:str, csvPath:str) -> int:
    activative_tensor, static_tensor = create_tensor_dict(model)
    nodeList, nodeDict = create_operator_list_dict(model, static_tensor)
    ##################################
    #         Topoligial Sort        #
    topo_order =  scheduler(nodeDict, nodeList)
    ##################################
    memoryTable = [{"valid":0, "address":0, "size":config.MEMORY_SIZE, "tensor":""}]
    init = [ tensor.name for tensor in model.graph.initializer]
    modelinputList = []
    for tensor in model.graph.input:
        if tensor.name not in init:
            modelinputList.append(tensor.name)
    del init
    memMAX = 0
    for inputName in modelinputList:
        inputSize = tool.operator_Mem_Bytes(activative_tensor[inputName])
        _, memoryTable = tool.malloc(inputName,inputSize,memoryTable)
    
    
    if os.path.isfile(csvPath):
        os.remove(csvPath)
    second = 0 
    memMAX = tool.dump_csv(csvPath, memoryTable,memMAX, second)
    
    
    last_use = {}   # tensorName -> last operator index
    first_use = {}
    for idx, op in enumerate(topo_order):
        name = nodeList[op]
        # all inputs get “last used” at idx
        for t in nodeDict[name]['input']:
            last_use[t] = idx
            if(not first_use.get(t)): # not an output? or it's an graph.input
                first_use.setdefault(t, idx)
        # outputs are “born” here; if never read, last_use == idx
        for t in nodeDict[name]['output']:
            last_use.setdefault(t, idx)
            first_use.setdefault(t, idx)
                
    tensor_lifetimes = {}  # tensor -> (start, end)
    
    for key in last_use:
        start = first_use[key]
        end = last_use[key]
        tensor_lifetimes[key] = (start, end)
        # print(f"tensor: {key} with lifetime begin: {start} and end: {end}")
    
    
    # First, collect tensor sizes
    tensor_sizes = {}
    tensor_size_set = set()
    for tensor_name in tensor_lifetimes:
        tensor_sizes[tensor_name] = tool.operator_Mem_Bytes(activative_tensor[tensor_name])
        tensor_size_set.add(tensor_sizes[tensor_name])
    
    # Determine the total range of time
    max_time = max(end for _, end in tensor_lifetimes.values())

    # Initialize time-based stats
    time_liveness_stats = []  # List of (timestamp, num_live_tensors, total_bytes_live)
    
    # 1. Quantize all tensor sizes to 100KB buckets (rounded up)
    QUANTIZE_UNIT = 100 * 1024  # 100 KB

    quantized_tensor_sizes = {}
    quantized_size_set = set()
    large_block_list = set()
    
    for tensor_name, size in tensor_sizes.items():
        # Round up to the nearest 100KB
        bucket_size = int(math.ceil(size / QUANTIZE_UNIT) * QUANTIZE_UNIT)
        quantized_tensor_sizes[tensor_name] = bucket_size
        quantized_size_set.add(bucket_size)

    quantized_size_set = sorted(quantized_size_set)

    # 2. Count per-timestamp how many of each quantized size are alive
    bucket_liveness_stats = []  # List of {timestamp -> {bucket_size: count}}

    for t in range(max_time + 1):
        bucket_count = defaultdict(int)
        for name, (start, end) in tensor_lifetimes.items():
            if start <= t <= end:
                b = quantized_tensor_sizes[name]
                if(b == quantized_size_set[-1]):
                    # add the name for large block list
                    large_block_list.add(name)
                    
                bucket_count[b] += 1
        bucket_liveness_stats.append((t, dict(bucket_count)))

    # 3. Print the report
    print("\n=== Bucketed Tensor Liveness Report (per 100KB) ===")
    for t, bucket_count in bucket_liveness_stats:
        print(f"Time {t:>2} :", end=" ")
        for b in sorted(bucket_count.keys()):
            print(f"{b//1024:>5}KB: {bucket_count[b]}", end="  ")
        print()
    
    # starting address
    largest_block_size = quantized_size_set[-1]

    '''
    Large block estimation
    '''
    tensor_overlaps_large = {}  # tensor_name -> True/False

    for t1, (start1, end1) in tensor_lifetimes.items():
        overlaps = False
        if(t1 in large_block_list):
            tensor_overlaps_large[t1] = overlaps
            continue
        for large_tensor in large_block_list:
            start2, end2 = tensor_lifetimes[large_tensor]
            # Check if lifetimes overlap
            # We only care about the lifetimes overlap when the it's will affect the placement for the large 
            # tensor 
            if (start1 < start2 and end2 > end1 and end1 > start2):
                overlaps = True
                break
        tensor_overlaps_large[t1] = overlaps


    for operator in topo_order:
        operatorName = nodeList[operator]
        inputList = nodeDict[operatorName]['input']
        initsList = nodeDict[operatorName]['initializer']
        outputList = nodeDict[operatorName]['output']
        # fullPath = os.path.join(operatorPath,operatorName+".onnx")    
        # second = reTimeSingleOperator(fullPath)
        second += 1
        for tensorName in outputList:
            memory = tool.operator_Mem_Bytes(activative_tensor[tensorName])
            _ , memoryTable = tool.malloc_lifetime_aware(
                tensorName,
                memory,
                memoryTable,
                largest_block_size,
                tensor_overlaps_large
            )
            '''_ , memoryTable = tool.malloc(
                tensorName,
                memory,
                memoryTable
            )'''
            
            tool.memory_table_check(memoryTable)
        memMAX = tool.dump_csv(csvPath,memoryTable, memMAX, second)
        for tensorName in inputList:
            activative_tensor[tensorName]['consumer'].remove(operatorName)
            if len(activative_tensor[tensorName]['consumer']) == 0:
                memoryTable = tool.free(tensorName, memoryTable) 

    return memMAX
            
    

    

    

    









