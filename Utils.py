# Topological sort untuk urutan menghitung grad pada proses backward
def topological_sort(root):
    visited = set()
    stack = []
    
    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        for child in node._previous:
            dfs(child)
        stack.append(node)
    
    dfs(root)
    return stack