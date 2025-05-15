from collections import defaultdict
 
# Class to represent a graph
class topoGraph:
    def __init__(self,vertices):
        self.graph = defaultdict(list)
        self.V = vertices #No. of vertices
 
    def addEdge(self,u,v):
        self.graph[u].append(v)
 
    def topologicalSortUtil(self,v,visited,stack):
 
        visited[v] = True
        for i in self.graph[v]:
            if visited[i] == False:
                self.topologicalSortUtil(i,visited,stack)
        stack.insert(0,v)
        
    def topologicalSort(self):
        visited = [False]*self.V
        stack =[]
        for i in range(self.V):
            if visited[i] == False:
                self.topologicalSortUtil(i,visited,stack)
        return stack
 
