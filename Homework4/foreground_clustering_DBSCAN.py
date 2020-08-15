import time
import numpy as np
import math
from collections import deque

import open3d as o3d
import matplotlib.pyplot as plt

from ground_detection_SVD import ground_detection_on3segs, plot_clusters, read_velodyne_bin, pcd_preprocessing


if __name__ == "__main__":
    path = "test/000099.bin"
    points = read_velodyne_bin(path)
    points = pcd_preprocessing(points)

    start = time.time()
    ground_idx, foreground_idx = ground_detection_on3segs(points)
    print("Ground detection takes {}ms".format(1000 * (time.time() - start)))

