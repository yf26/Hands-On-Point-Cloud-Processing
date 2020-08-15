# 文件功能： 实现 K-Means 算法

import numpy as np
from scipy import spatial


class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, n_clusters=2, tolerance=0.0001, max_iter=800):
        self.k_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter
        self.center_ = None

    def init_choice(self, data):
        """
        kmeans++ with different initialization
        :param data:
        :return:
        """
        result = []
        rng = np.random.default_rng()
        center_idx = int(rng.choice(data.shape[0], 1, replace=False))
        result.append(center_idx)
        while len(result) < self.k_:
            kdtree = spatial.KDTree(data[result, :])
            d, _ = kdtree.query(data, 1)
            dd = np.square(d)
            # dd = d
            distribution = np.divide(dd, np.sum(dd))
            center_idx = int(rng.choice(data.shape[0], 1, replace=False, p=distribution))
            result.append(center_idx)
        return result

    def fit(self, data):
        # 作业1
        # 屏蔽开始
        # rng = np.random.default_rng()
        # center_init_idx = rng.choice(data.shape[0], self.k_, replace=False)
        center_init_idx = self.init_choice(data)
        center = data[center_init_idx, :]
        converged = False
        count = 0
        while not converged and count <= self.max_iter_:
            count += 1
            # print(count)
            # print(center)
            kdtree = spatial.KDTree(center, 1)
            _, belonging = kdtree.query(data, 1)
            center_new = np.empty(center.shape)
            for i in range(self.k_):
                center_new[i] = np.mean(data[belonging == i, :], axis=0)
            if np.all((center_new - center) < self.tolerance_):
                converged = True
                self.center_ = center_new
            else:
                center = center_new
        # 屏蔽结束

    def predict(self, p_datas):
        # result = []
        # 作业2
        # 屏蔽开始
        if self.center_ is None:
            print("Fit model first!")
            return None

        kdtree = spatial.KDTree(self.center_, 1)
        _, result = kdtree.query(p_datas, 1)
        return result


if __name__ == '__main__':
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    k_means = K_Means(n_clusters=2)
    k_means.fit(x)
    cat = k_means.predict(x)
    print(cat)
