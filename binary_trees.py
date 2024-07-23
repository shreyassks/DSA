# Minimum time taken to burn the binary tree given the target leaf node

class TreeNode:
    def __init__(self, val=None):
        self.val = val
        self.left = None
        self.right = None


def helper(arr: list[int]):
    root = TreeNode(arr[0])
    queue = [root]
    i = 1
    while i < len(arr):
        node = queue.pop(0)
        if arr[i] is not None:
            node.left = TreeNode(arr[i])
            queue.append(node.left)
        i += 1
        if i < len(arr) and arr[i] is not None:
            node.right = TreeNode(arr[i])
            queue.append(node.right)
        i += 1
    return root


def deserialize(arr: list[int]):
    t = helper(arr)
    return t


def minimum_time_to_burn(root: TreeNode, target: TreeNode) -> int:
    parent = {root.val: None}
    queue = [root]

    while len(queue) != 0:
        node = queue.pop(0)
        if node.val == target.val:
            target = node
        if node.left:
            queue.append(node.left)
            parent[node.left.val] = node
            print(node.left.val, node.val)
        if node.right:
            queue.append(node.right)
            parent[node.right.val] = node
            print(node.right.val, node.val)

    queue = [target]
    visited = set()
    min_time = 0
    while len(queue) != 0:
        level_size = len(queue)
        for i in range(level_size):
            node = queue.pop(0)
            visited.add(node)
            if node.left and node.left not in visited:
                queue.append(node.left)
            if node.right and node.right not in visited:
                queue.append(node.right)
            if parent[node.val] and parent[node.val] not in visited:
                queue.append(parent[node.val])
        min_time += 1

    return min_time


# Plant disease spreading, find how much time needed to reach stability
def stability(root: TreeNode) -> int:
    queue = [root]
    cnt = 0
    while len(queue) != 0:
        level_size = len(queue)
        for i in range(level_size):
            node = queue.pop(0)
            if node.left and node.left.val > node.val:
                node.left.val -= node.val
                queue.append(node.left)
            if node.right and node.right.val > node.val:
                node.right.val -= node.val
                queue.append(node.right)
            if node.left and node.left.val < node.val and node.right and node.right.val < node.val:
                break
        cnt += 1
    return cnt - 1


# Sherlock Trap, Graph based
class Graph:
    def __init__(self, colors, k):
        self.graph = {}
        self.colors = colors
        self.k = k

    def add_edges(self, vertex_a, vertex_b, cost):
        if vertex_a not in self.graph:
            self.graph[vertex_a] = []
        self.graph[vertex_a].append((vertex_b, cost))

    def mapping_nodes_to_colours(self, colours):
        mapping = {}
        keys = sorted(self.graph.keys())
        for idx, (node, tag) in enumerate(zip(keys, colours)):
            if tag == 0:
                mapping[node] = "Blue"
            else:
                mapping[node] = "Red"
        return mapping

    def trap_condition(self, path) -> bool:
        num_blue, num_red = 0, 0
        maps = self.mapping_nodes_to_colours(self.colors)
        if not path:
            return False
        for node, _ in path[0]:
            if maps[node] == "Red":
                num_red += 1
            else:
                num_blue += 1
        if abs(num_blue - num_red) <= self.k:
            return True
        else:
            return False

    def find_all_paths(self, start, end, path, cost=0):
        path = path + [(start, cost)]
        if start == end:
            return [path]
        if start not in self.graph:
            return []
        paths = []
        for node, new_cost in self.graph[start]:
            if node not in path:
                new_path = self.find_all_paths(node, end, path, cost + new_cost)
                condition = self.trap_condition(new_path)
                if condition:
                    paths.extend(new_path)
        return paths

    @staticmethod
    def minimum_time_to_exit(paths, target) -> int:
        if paths:
            tuples = [(node, cost) for path in paths for node, cost in path if node == target]
            return min([t[-1] for t in tuples])
        else:
            return -1


'''
Given a binary tree. Find the path of the node having lowest value to the node having highest value.
If the tree has nodes with value 1,2,3,4,5,6,7, 8, 9, 10
return the nodes in the path between 1 to 10.
'''

lowest, highest = 100000, float("-inf")


def find_minimum_node(root: TreeNode):
    global lowest
    if root is None:
        return -1
    if root.val < lowest:
        lowest = min(lowest, root.val)
    find_minimum_node(root.left)
    find_minimum_node(root.right)


def find_maximum_node(root: TreeNode):
    global highest
    if root is None:
        return -1
    if root.val > highest:
        highest = max(highest, root.val)
    find_maximum_node(root.left)
    find_maximum_node(root.right)


def lowest_common_ancestor(root: TreeNode, p: int, q: int):
    if root is None:
        return None
    if root.val == p or root.val == q:
        return root
    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)

    if left is not None and right is not None:
        return root
    if left is None:
        return right
    else:
        return left


def path_between_two_nodes(root: TreeNode, lca: TreeNode, node1: int, node2: int) -> list:
    parent = {}
    queue = [root]
    while len(queue) != 0:
        node = queue.pop(0)
        if node.left:
            queue.append(node.left)
            parent[node.left.val] = node
        if node.right:
            queue.append(node.right)
            parent[node.right.val] = node
    print(parent)

    connecting_nodes = []
    node = node1
    while parent[node] != lca:
        new_node = parent[node]
        connecting_nodes.append(new_node.val)
        node = new_node.val
    connecting_nodes.append(lca.val)
    node = node2
    while parent[node] != lca:
        new_node = parent[node]
        connecting_nodes.append(new_node.val)
        node = new_node.val
    return connecting_nodes


'''
Given an array of billion of numbers. Billions of queries are generated with parameters as starting and an ending index. 
Both these indices lie within that array. 
Find the maximum number between these two indices in less than O(N)
'''


class SegmentNode:
    def __init__(self, left: int, right: int):
        self.data = None
        self.start_interval = left
        self.end_interval = right
        self.left = None
        self.right = None


def construct_tree(arr: list[int], start: int, end: int) -> SegmentNode:
    if start == end:
        node = SegmentNode(start, end)
        node.data = arr[start]
        return node

    node = SegmentNode(start, end)
    mid = (start + end) // 2

    node.left = construct_tree(arr, start, mid)
    node.right = construct_tree(arr, mid+1, end)

    node.data = max(node.left.data , node.right.data)
    return node


def maximum_in_range(root: SegmentNode, start: int, end: int) -> int:
    if root.start_interval > end or root.end_interval < start:
        return 0
    elif root.start_interval >= start and root.end_interval <= end:
        return root.data
    else:
        return max(maximum_in_range(root.left, start, end), maximum_in_range(root.right, start, end))


if __name__ == "__main__":
    root_array = [3, 5, 1, 6, 2, 0, 8, None, None, 7, 4]
    # root_array = [6, 7, 5, 1, None, None, None, 4, 2, 3, None]
    tree = deserialize(root_array)
    # print(tree.val)
    # print(tree.left.val)
    # print(tree.right.val)
    # print(tree.left.left.val)
    # print(tree.left.left.left.val)
    # print(tree.left.left.right.val)
    # print(tree.left.left.left.left.val)

    # target_node = TreeNode(7)
    # prob1 = minimum_time_to_burn(tree, target_node)
    # print(f"Minimum time needed to burn the tree from target node - {prob1} timestamps\n")

    # array = [6, 8, 7, 3, None, None, 4, 1, None]
    # tree2 = deserialize(array)
    # prob2 = stability(tree2)
    # print(f"Time taken to attain stability - {prob2} timestamps")

    # nodes = [[2, 3, 1], [1, 2, 1], [4, 3, 2], [5, 6, 2], [1, 4, 2], [3, 5, 2]]
    # color_codes = [1, 0, 1, 0, 1, 0]
    # limit = 2
    # graph = Graph(colors=color_codes, k=limit)
    # for u, v, c in nodes:
    #     graph.add_edges(u, v, c)
    # graph.add_edges(6, 0, 0)
    #
    # all_paths = graph.find_all_paths(1, 6, [])
    # print(f"All paths to exit satisfying the condition - {all_paths}")
    # min_time = graph.minimum_time_to_exit(all_paths, target=6)
    # print(f"Minimum time to exit from the trap - {min_time}")

    # find_minimum_node(tree)
    # print(lowest)
    # find_maximum_node(tree)
    # print(highest)
    # lca = lowest_common_ancestor(tree, 0, 8)
    # print(lca)
    # c_nodes = path_between_two_nodes(tree, lca, 0, 8)
    # print(c_nodes)

    # array = [3, 8, 6, 7, -2, -8, 4, 9]
    # seg_tree = construct_tree(array, 0, len(array)-1)
    # print(f"Maximum value for entire array is - {seg_tree.data}")
    # max_val = maximum_in_range(seg_tree, 2, 6)
    # print(f"Maximum value in the given range is - {max_val}")

    import re
    f = "i want to pluck you at 1990/12/12"
    print(re.sub(r'\b\d{4}/\d{1,2}/\d{1,2}\b', "", f))
