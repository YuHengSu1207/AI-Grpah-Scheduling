from . import *
from .anal import anal
from typing import Optional
from src.cost.anal import anal
from src.management import tool
from src.structure import DATA_SIZE_DTYPE
from collections import defaultdict

def get_tensor_use_count(model: onnx.ModelProto):
    use_count = defaultdict(int)
    for node in model.graph.node:
        for inp in node.input:
            use_count[inp] += 1
    return use_count


def costFunction(model:onnx.ModelProto,layout:str,  memoryTable:Optional[list] = None, csvPath:Optional[str] = None):
    
    use_count = get_tensor_use_count(model)
    
    memoryRequest = 0
    for tensor in model.graph.input:
        dims = [memory.dim_value for memory in tensor.type.tensor_type.shape.dim]
        memory = DATA_SIZE_DTYPE[tensor.type.tensor_type.elem_type]
        for dim in dims:
            memory *= dim
        request, memoryTable =  tool.malloc(tensor.name, memory // 8, memoryTable)
        memoryRequest += request
    # for tensor in model.graph.initializer:
    #     dims = tensor.dims
    #     memory = DATA_SIZE_DTYPE[tensor.data_type]
    #     for dim in dims:
    #         memory *= dim
    #     request, memoryTable =  tool.malloc(tensor.name, memory // 8, memoryTable)
    #     memoryRequest += request 


    cycle = 0
    # dump the csv elsewhere
    nodes = [node for node in model.graph.node]
    
    for node in nodes:
        # print(f" ======== begin of node {node.name} ========")
        # print(f"memory table : {memoryTable} and cycle {cycle}")
        '''
        Add initializer at the beginning
        '''
        for input in node.input:
            is_initializer = False
            initializer = None
            initializer_name = None
            tensor_name = input
            for tensor_info, output in enumerate(model.graph.initializer):
                if  output.name == tensor_name:
                    initializer = output
                    initializer_name = tensor_name
                    is_initializer = True
                    break
                
            if(is_initializer):
                typeW = initializer.data_type
                staticMemX = DATA_SIZE_DTYPE[typeW]
                for dim in initializer.dims:
                    staticMemX *= dim
                request, memoryTable = tool.malloc(initializer_name , staticMemX // 8 , memoryTable)
        
        # dump the first time here
        if(cycle == 0):
            _ = tool.dump_csv(csvPath, memoryTable, config.MEMORY_SIZE, 0)
        '''
        Do the analysis
        '''
        _, table ,memoryTable = anal.analyticalModel(
            model=model,
            layout=layout,
            node=node,
            memoryTable=memoryTable,
            csvPath=csvPath,
            previous_cycle=cycle,
            use_count=use_count
        )
        cycle += table['cycle']
        
        # print(f"memory table : {memoryTable} and cycle {cycle}")
        # print(f" ======== end of node {node.name} ========")
        
        # memoryRequest += request
    _ = tool.dump_csv(csvPath, memoryTable, config.MEMORY_SIZE, 0)
    return cycle