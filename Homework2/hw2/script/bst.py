import math
import numpy as np
import time
from script.result_set import KNNResultSet, RadiusNNResultSet


class Node:
    def __init__(self, key, value=-1):
        self.left = None
        self.right = None
        self.key = key
        self.value = value

    def __str__(self):
        return "Node key: {}, value: {}".format(self.key, self.value)


def insert(root, key, value=-1):
    if root is None:
        root = Node(key, value)
    else:
        if key < root.key:
            root.left = insert(root.left, key, value)
        elif key > root.key:
            root.right = insert(root.right, key, value)
        else:
            print("Node already exist!")
            pass
    return root


def inorder(root):
    if root is not None:
        inorder(root.left)
        print(root)
        inorder(root.right)


def preorder(root):
    if root is not None:
        print(root)
        preorder(root.left)
        preorder(root.right)


def postorder(root):
    if root is not None:
        print(root)
        postorder(root.left)
        postorder(root.right)


def search_recursive(root: Node, key):
    if root is None or root.key == key:
        return root
    else:
        if key < root.key:
            return search_recursive(root.left, key)
        elif key > root.key:
            return search_recursive(root.right, key)


def search_iterative(root: Node, key):
    current = root
    while current is not None:
        if key == current.key:
            return current
        elif key < current.key:
            current = current.left
        elif key > current.key:
            current = current.right
    return current


def knn_search(root: Node, result_set: KNNResultSet, query_key):
    if root is None:
        return False

    result_set.add_point(math.fabs(root.key - query_key), root.value)
    if result_set.get_worst_dist() == 0:
        return True

    if query_key <= root.key:
        if knn_search(root.left, result_set, query_key):
            return True
        elif math.fabs(root.key - query_key) < result_set.get_worst_dist():
            return knn_search(root.right, result_set, query_key)
        return False

    if query_key > root.key:
        if knn_search(root.right, result_set, query_key):
            return True
        elif math.fabs(root.key - query_key) < result_set.get_worst_dist():
            return knn_search(root.left, result_set, query_key)
        return False


def radius_search(root: Node, result_set: RadiusNNResultSet, query_key):
    if root is None:
        return False

    result_set.add_point(math.fabs(root.key - query_key), root.value)

    if query_key <= root.key:
        if radius_search(root.left, result_set, query_key):
            return True
        elif math.fabs(root.key - query_key) < result_set.get_worst_dist():
            return radius_search(root.right, result_set, query_key)
        return False

    if query_key > root.key:
        if radius_search(root.right, result_set, query_key):
            return True
        elif math.fabs(root.key - query_key) < result_set.get_worst_dist():
            return radius_search(root.left, result_set, query_key)
        return False


if __name__ == "__main__":
    data_size = 100000
    data = np.random.permutation(data_size)
    print(data)
    print("Data size = {}".format(data_size))

    # root = None
    # start = time.time()
    # for i in range(data.size):
    #     root = insert(root, key=data[i], value=i)
    # print("Build tree takes {}ms".format(1000 * (time.time() - start)))
    # print("")
    #
    # # print("Inorder traversing: ")
    # # inorder(root)
    # # print("")
    #
    # search_key = -5
    # start = time.time()
    # print("Recursively search key {}: \n{}".format(search_key, search_recursive(root, search_key)))
    # print("Search takes {}ms".format(1000 * (time.time() - start)))
    # print("")
    #
    # start = time.time()
    # print("Iteratively search key {}: \n{}".format(search_key, search_iterative(root, search_key)))
    # print("Search takes {}ms".format(1000 * (time.time() - start)))
    # print("")

    root = None
    for i in range(len(data)):
        root = insert(root, key=data[i], value=i)
    query_data = 4.2

    # print(search_recursive(root, query_data))

    print("************KNN Search*************")
    K = 3
    start = time.time()
    knn_result_set = KNNResultSet(K)
    knn_search(root, knn_result_set, query_data)
    print("KNN Search takes {}ms".format(1000 * (time.time() - start)))
    print("Comparision times = {}\n".format(knn_result_set.comparision_count))

    knn_result_set.list()

    # for i in range(K):
    #     print(data[knn_result_set.dist_index_list[i].index])

    print("***********Radius Search************")
    R = 1.21
    start = time.time()
    radius_result_set = RadiusNNResultSet(R)
    radius_search(root, radius_result_set, query_data)
    print("Radius Search takes {}ms".format(1000 * (time.time() - start)))
    print("Comparision times = {}\n".format(radius_result_set.comparision_count))

    radius_result_set.list()

    # for i in range(radius_result_set.size()):
    #     print(data[radius_result_set.dist_index_list[i].index])
