# 文件功能：实现 GMM 算法

import numpy as np
import scipy
import pylab
import random,math
from scipy import spatial

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
plt.style.use('seaborn')


class GMM(object):
    def __init__(self, n_clusters, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.model_params = None
        self.init_center = None

    def init_choice(self, data):
        """
        kmeans++ initialization
        :param data:
        :return:
        """
        result = []
        rng = np.random.default_rng()
        center_idx = int(rng.choice(data.shape[0], 1, replace=False))
        result.append(center_idx)
        while len(result) < self.n_clusters:
            kdtree = spatial.KDTree(data[result, :])
            d, _ = kdtree.query(data, 1)

            mean_d = np.mean(d, axis=0)
            dd = []
            for dist in d:
                if dist < 1.25 * mean_d:
                    dd.append(0)
                else:
                    dd.append(np.exp(dist))

            distribution = np.divide(dd, np.sum(dd))
            center_idx = int(rng.choice(data.shape[0], 1, replace=False, p=distribution))
            result.append(center_idx)
        return result

    # 屏蔽开始
    # 计算权重
    def posterior(self, data, mean_k, cov_k, pi_k):
        result = []
        for mean, cov, pi in zip(mean_k, cov_k, pi_k):
            temp = []
            gaussian = scipy.stats.multivariate_normal.pdf(mean=mean, cov=cov)
            for i in range(data.shape[0]):
                x_n = data[i]
                temp.append(pi * gaussian.pdf(x_n))
            temp = np.asarray(temp)
            result.append(temp)
        post_temp = np.asarray(result).T
        post_sum = np.sum(post_temp, axis=1)
        post = np.array([post_temp[i] / post_sum[i] for i in range(post_sum.shape[0])])
        return post

    def EM(self, data, mean_k, cov_k, pi_k):
        post = self.posterior(data, mean_k, cov_k, pi_k)
        N_k = np.sum(post, axis=0)
        # 更新pi
        pi_k_new = N_k / data.shape[0]
        
        # 更新mean
        mean_k_new = [[] for i in range(self.n_clusters)]
        for k in range(N_k.shape[0]):
            mean_k_new[k] = list(np.sum(np.expand_dims(post[:, k], axis=1) * data, axis=0) / N_k[k])  # TODO

        # 更新cov
        cov_k_new = [[] for i in range(self.n_clusters)]
        for k in range(N_k.shape[0]):
            cov_k_temp = 0
            for n in range(data.shape[0]):
                diff = np.expand_dims(data[n] - mean_k_new[k], axis=1)
                cov_k_temp += post[n, k] * diff * diff.T
            cov_k_new[k] = cov_k_temp / N_k[k]

        return np.array(mean_k_new), np.array(cov_k_new), pi_k_new
    # 屏蔽结束
    
    def fit(self, data):
        # 作业3
        # 屏蔽开始
        eps = 1e-4
        amplitude = 0.3

        rng = np.random.default_rng()
        # mean_idx = rng.choice(data.shape[0], self.n_clusters, replace=False)
        mean_idx = self.init_choice(data)
        mean_k = data[mean_idx]

        ###
        self.init_center = mean_k
        ###

        cov_k = [amplitude*np.identity(data.shape[1]) for i in range(self.n_clusters)]
        pi_k = [1/self.n_clusters for i in range(self.n_clusters)]

        count = 0
        converged = False
        while not converged and count < self.max_iter:
            count += 1
            if count % 5 == 0:
                print("GMM Iteration ", count)
            mean_k_new, cov_k_new, pi_k_new = self.EM(data, mean_k, cov_k, pi_k)

            # TODO to avoid singular point, how to detect ill posed position
            for k in range(self.n_clusters):
                if np.linalg.norm(cov_k_new[k]) < 0.01:
                    cov_k_new[k] = amplitude * np.identity(data.shape[1])
                    mean_k_new[k] = data[rng.choice(data.shape[0], 1)]

            # print(mean_k_new)
            # print(mean_k)
            diff_mean_max = np.max(np.fabs(mean_k_new - mean_k))
            diff_cov_max = np.max(np.fabs(cov_k_new - cov_k))
            diff_pi_max = np.max(np.fabs(pi_k_new - pi_k))
            if diff_cov_max < eps and diff_mean_max < eps and diff_pi_max < eps or count == self.max_iter:
                converged = True
                self.model_params = (mean_k_new, cov_k_new, pi_k_new)
            else:
                mean_k, cov_k, pi_k = mean_k_new, cov_k_new, pi_k_new

        # 屏蔽结束
    
    def predict(self, data):
        # 屏蔽开始
        post = self.posterior(data, self.model_params[0], self.model_params[1], self.model_params[2])
        belonging = np.argmax(post, axis=1)
        return belonging
        # 屏蔽结束


# 生成仿真数据
def generate_X(true_Mu, true_Var):
    # 第一簇的数据
    num1, mu1, var1 = 400, true_Mu[0], true_Var[0]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)
    # 第二簇的数据
    num2, mu2, var2 = 600, true_Mu[1], true_Var[1]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
    # 第三簇的数据
    num3, mu3, var3 = 1000, true_Mu[2], true_Var[2]
    X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)
    # 合并在一起
    X = np.vstack((X1, X2, X3))
    # 显示数据
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X1[:, 0], X1[:, 1], s=5)
    plt.scatter(X2[:, 0], X2[:, 1], s=5)
    plt.scatter(X3[:, 0], X3[:, 1], s=5)
    plt.show()
    return X


if __name__ == '__main__':
    # 生成数据
    true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    true_Var = [[1, 3], [2, 2], [6, 2]]
    X = generate_X(true_Mu, true_Var)

    gmm = GMM(n_clusters=3)
    gmm.fit(X)
    print(gmm.model_params)
    cat = gmm.predict(X)
    print(cat)
    # 初始化

    

