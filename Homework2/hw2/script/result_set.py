import copy


class DistIndex:
    def __init__(self, distance, index):
        self.distance = distance
        self.index = index

    def __lt__(self, other):  # less than
        return self.distance < other.distance

    def __str__(self):
        return "Distance = {}, Index = {}".format(self.distance, self.index)


class KNNResultSet:
    def __init__(self, capacity):
        self.capacity = capacity
        self.count = 0
        self.worst_dist = 1e10
        self.dist_index_list = []

        for i in range(capacity):
            self.dist_index_list.append(DistIndex(self.worst_dist, 0))

        self.comparision_count = 0

    def size(self):
        return self.count

    def full(self):
        return self.count == self.capacity

    def get_worst_dist(self):
        return self.worst_dist

    def list(self):
        print("Distance-Index list:")
        for i in range(self.capacity):
            print(self.dist_index_list[i])

    def add_point(self, dist, index):
        self.comparision_count += 1

        if dist > self.worst_dist:
            return

        if self.count < self.capacity:
            self.count += 1

        current_pos = self.count - 1
        while current_pos > 0:
            if self.dist_index_list[current_pos-1].distance > dist:
                self.dist_index_list[current_pos] = copy.copy(self.dist_index_list[current_pos-1])
                current_pos -= 1
            else:
                break
            # print(current_pos + 1)
        self.dist_index_list[current_pos].distance = dist
        self.dist_index_list[current_pos].index = index
        self.worst_dist = self.dist_index_list[self.capacity - 1].distance


class RadiusNNResultSet:
    def __init__(self, radius):
        self.radius = radius
        self.count = 0
        self.worst_dist = radius
        self.dist_index_list = []
        self.comparision_count = 0

    def get_worst_dist(self):
        return self.worst_dist

    def size(self):
        return self.count

    def list(self):
        print("Distance-Index list:")
        for i in range(self.count):
            print(self.dist_index_list[i])

    def add_point(self, distance, index):
        self.comparision_count += 1
        if distance <= self.worst_dist:
            self.dist_index_list.append(DistIndex(distance, index))
            self.count += 1
        else:
            return


if __name__ == "__main__":
    test_set = KNNResultSet(3)
    test_set.add_point(2, 1)
    test_set.add_point(3, 0)
    test_set.add_point(1, 0)
    # test_set.add_point(2, 1)
    test_set.add_point(4, 0)
    test_set.add_point(3, 0)
    test_set.list()
    print(test_set.get_worst_dist())
