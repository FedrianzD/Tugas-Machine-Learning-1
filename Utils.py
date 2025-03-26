# Topological sort untuk urutan menghitung gradient pada proses backward
def topological_sort(root):
    visited = set()
    stack = []
    order = []
    to_visit = [root]

    while to_visit:
        node = to_visit.pop()
        if node not in visited:
            visited.add(node)
            for child in node._previous:
                to_visit.append(child)
            order.append(node)

    return order