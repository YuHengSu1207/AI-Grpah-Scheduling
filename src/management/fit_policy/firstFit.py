def fit(memoryTable:list,memory:int,memory_MAX:int)->int:
    fit_index = -1
    for cnt in range(len(memoryTable)):
        block = memoryTable[cnt]
        if (block['valid'] == 1):
            continue
        else:
            if block['size'] >= memory:
                fit_index = cnt
                break
    return fit_index 