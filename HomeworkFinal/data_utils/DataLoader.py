import numpy as np
import warnings
import os
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    # m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    # pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class MyKittiDataLoader(Dataset):
    def __init__(self, root, cache, npoint=256, split='train', normal_channel=False, cache_size=45000):
        self.root = root
        self.npoints = npoint
        self.catfile = '/disk/users/sc468/workspace/HomeworkFinal/dataset_info/mykitti_cls_names.txt'

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.normal_channel = normal_channel

        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open('/disk/users/sc468/workspace/HomeworkFinal/dataset_info/mykitti_train.txt')]
        shape_ids['test'] = [line.rstrip() for line in open('/disk/users/sc468/workspace/HomeworkFinal/dataset_info/mykitti_test.txt')]

        assert (split == 'train' or split == 'test')
        shape_names = [x.split('_')[0] for x in shape_ids[split]]
        # list of (shape_name, shape_txt_file_path) tuple

        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i])) for i
                         in range(len(shape_ids[split]))]
        # print(self.datapath)
        print('The size of %s data is %d' % (split, len(self.datapath)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = cache  # from index to (point_set, cls) tuple

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if index in self.cache:
        # if False:
            # print("Load data from cache!")
            point_set, cls = self.cache[index]
        else:
            # start = time.time()
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32)

            point_set = np.fromfile(fn[1], dtype=np.float32, count=-1).reshape(-1, 3)

            # point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            # sample_start = time.time()

            point_number = point_set.shape[0]
            if point_number > self.npoints:
                # select npoints using FPS
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                # select npoints randomly
                append_choice = np.random.choice(point_number, self.npoints - point_number, replace=True)
                point_append = point_set[append_choice, :]
                point_set = np.append(point_set, point_append, axis=0)

            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

            if not self.normal_channel:
                point_set = point_set[:, 0:3]

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)
                # print("Push data to cache!")

            # sample_end = time.time()
            # print("IO time {:.4f}ms, preprocessing time {:.4f}ms".format(1000*(time.time() - start), 1000*(sample_end - sample_start)))
        return point_set, cls

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    import torch
    cache = {}
    data = MyKittiDataLoader('/disk/users/sc468/no_backup/my_kitti', cache, split='train',
                             normal_channel=False)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)
