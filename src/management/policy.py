from .fit_policy import bestFit, firstFit, newFit

def fit(memoryTable:list,memory:int,memory_MAX:int, fit_policy:str)->int:
    fit_list = {
        "bestFit" : bestFit.fit, 
        "firstFit": firstFit.fit,
        "newFit": newFit.fit
        }
    if fit_policy not in fit_list.keys():
        fit_policy = "bestFit"
    return fit_list[fit_policy](memoryTable=memoryTable,memory=memory,memory_MAX=memory_MAX)
