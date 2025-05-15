
import onnx 
from src.scheduler.scheduler import scheduler
from src.tool.util import create_operator_list_dict, create_tensor_dict
from src.management import tool
from src.structure import DATA_SIZE_DTYPE
import os
import src.config as config
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
            _, memoryTable = tool.malloc(tensorName, memory, memoryTable)
        memMAX = tool.dump_csv(csvPath,memoryTable, memMAX, second)
        for tensorName in inputList:
            activative_tensor[tensorName]['consumer'].remove(operatorName)
            if len(activative_tensor[tensorName]['consumer']) == 0:
                memoryTable = tool.free(tensorName, memoryTable) 

    return memMAX
            
    

    

    

    









