import numpy as np
import matplotlib.pyplot as plt

files = [
    "aniso.txt",
    "blobs.txt",
    "circle.txt",
    "moons.txt",
    "varied.txt"
]

colors = np.array(['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00'])

plt.figure(figsize=(10, 2))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)

plt_num = 1

for file in files:
    points = []
    with open("../data/" + file, "r") as f:
        for i in range(1500):
            str_list = f.readline().split(",")
            points.append([float(i) for i in str_list])
        points = np.array(points)

    result = []
    with open("../result/predict_" + file, "r") as f:
        for i in range(1500):
            belonging = int(f.readline())
            result.append(belonging)
        result = np.array(result)

    plt.subplot(1, 5, plt_num)
    plt.title(file, size=10)
    plt.scatter(points[:, 0], points[:, 1], s=10, color=colors[result])
    plt.xticks(())
    plt.yticks(())
    plt_num += 1

plt.show()
