
from src.structure import DATA_SIZE_DTYPE
from src.config import MEMORY_SIZE
from . import policy 
import csv
import os
# memoryTable = [{"valid":0, "address":0, "size":4194304, "tensor":""}]


from typing import List, Tuple, Dict

def malloc_lifetime_aware(
    tensorName: str,
    tensorSize: int,
    memoryTable: List[dict],
    largest_block_size: int,
    tensor_overlaps_large: dict
) -> Tuple[int, List[dict]]:
    """
    Allocate memory using best-fit, but prefer placing long-lived tensors at low addresses
    and bias shorter tensors upward to avoid overlap.
    """
    # fit policy
    pivot = -1
    bestRecord = MEMORY_SIZE + 1
    able_to_fit_at_low_addr = False
    low_addr_pivot = -1
    for cnt in range(len(memoryTable)):
        block = memoryTable[cnt]
        if (block['valid'] == 1):
            continue
        else:
            if not tensor_overlaps_large[tensorName]:
                # if not tensor overlap with large, make it normal
                if block['size'] >= tensorSize and block['size'] < bestRecord:
                    pivot = cnt
                    bestRecord = block['size']
            else:
                # if able to assign in the higher address space, make it fit in higher address spce
                if block['size'] >= tensorSize and block['size'] < bestRecord and block['address'] >= largest_block_size:
                    pivot = cnt
                    bestRecord = block['size']
                if block['size'] >= tensorSize and block['address'] < largest_block_size:
                    # if not able to fit in higher address space, make it the lower address pivot
                    able_to_fit_at_low_addr = True
                    low_addr_pivot = cnt

    if pivot == -1 and not able_to_fit_at_low_addr:
        print(f"Out of Memory : {tensorName} : {tensorSize/1024} KB")
        if memoryTable[-1]['valid'] == 1:
            return tensorSize, memoryTable
        else:  
            return tensorSize - memoryTable[-1]["size"], memoryTable
    else:
        if(not tensor_overlaps_large[tensorName]):
            oBlk = memoryTable[pivot]
            nBlk = {"valid":1, "address":oBlk['address'], "size":tensorSize, "tensor": tensorName}
            memoryTable[pivot]["address"] = oBlk['address'] + tensorSize
            memoryTable[pivot]["size"]    = oBlk['size'] - tensorSize
            memoryTable.insert(pivot,nBlk)
        else:
            # overlap with higher address space
            if pivot != -1: # fit in higher address space
                oBlk = memoryTable[pivot]
                nBlk = {"valid":1, "address":oBlk['address'], "size":tensorSize, "tensor": tensorName}
                memoryTable[pivot]["address"] = oBlk['address'] + tensorSize
                memoryTable[pivot]["size"]    = oBlk['size'] - tensorSize
                memoryTable.insert(pivot,nBlk)
            else:
                # should be able to fit at lower address
                assert(able_to_fit_at_low_addr)
                oBlk = memoryTable[low_addr_pivot]
                o_start = oBlk['address']
                o_size = oBlk['size']
                
                # Determine where to place the tensor so it stays above `largest_block_size`
                shifted_addr = largest_block_size
                assert shifted_addr + tensorSize <= o_start + o_size, f"Not enough room to fit tensor {tensorName} after shifting."

                # Reserve the space below `largest_block_size` (keep it valid=0)
                reserved_size = shifted_addr - o_start
                memoryTable[low_addr_pivot] = {
                    "valid": 0,
                    "address": o_start,
                    "size": reserved_size,
                    "tensor": ""
                }

                # Insert the new tensor allocation block right after the reserved area
                newBlock = {
                    "valid": 1,
                    "address": shifted_addr,
                    "size": tensorSize,
                    "tensor": tensorName
                }

                remaining_size = o_size - (reserved_size + tensorSize)
                if remaining_size > 0:
                    # If there's remaining space *after* the new block, insert it too
                    afterBlock = {
                        "valid": 0,
                        "address": shifted_addr + tensorSize,
                        "size": remaining_size,
                        "tensor": ""
                    }
                    memoryTable.insert(low_addr_pivot + 1, newBlock)
                    memoryTable.insert(low_addr_pivot + 2, afterBlock)
                else:
                    memoryTable.insert(low_addr_pivot + 1, newBlock)

                print(f"Assigning the tensor {tensorName} in lower space")
                print(memoryTable)
                    
        return 0, memoryTable

def memory_table_check(memoryTable: List[dict]):
    prev_addr = -1
    for cnt in range(len(memoryTable)):
        block = memoryTable[cnt]
        if(prev_addr > block["address"]):
            raise "memory address should be monotoneous increasing and none-overlap"
        prev_addr = block["address"] + block["size"]
    
    return


def malloc(tensorName:str, tensorSize:int, memoryTable:list)->tuple([int,list]):
    pivot = policy.fit(memoryTable, tensorSize, MEMORY_SIZE + 1, fit_policy="newFit") # 32MB + 1
    if pivot == -1:
        print(f"Out of Memory : {tensorName} : {tensorSize/1024} KB")
        if memoryTable[-1]['valid'] == 1:
            return tensorSize, memoryTable
        else:  
            return tensorSize - memoryTable[-1]["size"], memoryTable
    else:
        oBlk = memoryTable[pivot]
        nBlk = {"valid":1, "address":oBlk['address'], "size":tensorSize, "tensor": tensorName}
        memoryTable[pivot]["address"] = oBlk['address'] + tensorSize
        memoryTable[pivot]["size"]    = oBlk['size'] - tensorSize
        memoryTable.insert(pivot,nBlk)
        return 0, memoryTable

def free(tensorName:str,  memoryTable:list)->list:
    # add finding assert
    find_tensor = False
    for cnt in range(len(memoryTable)):
        if memoryTable[cnt]['tensor'] == tensorName:
            find_tensor = True
            block =  memoryTable[cnt]
            if cnt > 0 and cnt < (len(memoryTable) - 1): # 1 ~ (n-1)
                if memoryTable[cnt + 1]['valid'] == 0 and memoryTable[cnt - 1]['valid'] == 0:  # merge [cnt-1 cnt cnt+1] block
                    memoryTable[cnt - 1]['size'] += (block['size'] + memoryTable[cnt + 1]['size'])
                    memoryTable.remove(memoryTable[cnt + 1])
                    memoryTable.remove(block)
                elif memoryTable[cnt + 1]['valid'] != 0 and memoryTable[cnt - 1]['valid'] == 0: # merge [cnt-1 cnt] block
                    memoryTable[cnt - 1]['size'] += block['size']
                    memoryTable.remove(block)
                elif memoryTable[cnt + 1]['valid'] == 0 and memoryTable[cnt - 1]['valid'] != 0: # merge [cnt-1 cnt] block
                    memoryTable[cnt + 1]['address'] = block['address']
                    memoryTable[cnt + 1]['size'] += block['size']
                    memoryTable.remove(block)
                else:
                    memoryTable[cnt]['valid'] = 0
                    memoryTable[cnt]['tensor'] = ""
            elif cnt == 0 and cnt < len(memoryTable): # 0
                if memoryTable[cnt + 1]['valid'] == 0:
                    memoryTable[cnt + 1]['address'] = block['address']
                    memoryTable[cnt + 1]['size'] += block['size']
                    memoryTable.remove(block)
                else:
                    memoryTable[cnt]['valid'] = 0
                    memoryTable[cnt]['tensor'] = ""
            elif cnt > 0 and cnt == (len(memoryTable) - 1): # -1
                if memoryTable[cnt - 1]['valid'] == 0:
                    memoryTable[cnt - 1]['size'] += block['size']
                    memoryTable.remove(block)
                else:
                    memoryTable[cnt]['valid'] = 0
                    memoryTable[cnt]['tensor'] = ""
            break
        
    if not find_tensor:
        print(f"No found : {tensorName}")
        raise "Invalid free"
    
    return memoryTable


def dump_csv(csvPath:str,memoryTable:dict,memMAX:int,second:int) -> int:
    
    with open(csvPath, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            row = [second]
            for m in memoryTable:
                if m['valid'] == 0 and m['tensor'] == '':
                    row.append('empty')
                    row.append(m['address'])
                    row.append(m['address'] + m['size'])
                else:
                    row.append(m['tensor'])
                    row.append(m['address']) 
                    row.append(m['address'] + m['size'])
                    if m['address'] + m['size'] > memMAX: memMAX = m['address'] + m['size']
            writer.writerow(row)
    return memMAX

def operator_Mem_Bytes(tensorInfo):
    memory = 1
    for dim in tensorInfo['dims']: memory *= dim
    return memory * DATA_SIZE_DTYPE[tensorInfo['element_type']] // 8 



