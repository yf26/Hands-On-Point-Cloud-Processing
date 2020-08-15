import numpy as np
import os
from matplotlib import pyplot as plt

my_db_path = "/disk/users/sc468/no_backup/my_kitti/"
car_files = os.listdir(os.path.join(my_db_path, "Car"))
cyclist_files = os.listdir(os.path.join(my_db_path, "Cyclist"))
pedestrian_files = os.listdir(os.path.join(my_db_path, "Pedestrian"))
dontcare_files = os.listdir(os.path.join(my_db_path, "DontCare"))

labels = ["Car", "Cyclist", "Pedestrian", "DontCare"]

plt.figure(1, figsize=(6, 6))
sizes = [len(car_files), len(cyclist_files), len(pedestrian_files), len(dontcare_files)]
colors = ['magenta', 'yellowgreen', 'lightskyblue', 'yellow']
plt.pie(
    sizes, labels=labels, colors=colors, autopct='%3.2f%%', shadow=False, startangle=90, pctdistance=0.6
)
# fig1.axis('equal')
plt.show()


all_files = [car_files, cyclist_files, pedestrian_files, dontcare_files]
plt.figure(2, figsize=(12, 8))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                     hspace=.01)

num_plt = 1
for files, cls in zip(all_files, labels):
    print(cls)
    num_points_list = []
    for file in files:
        file_path = os.path.join(str(my_db_path), str(cls), str(file))
        points = np.fromfile(file_path, dtype=np.float32, count=-1).reshape(-1, 3)
        num_points_list.append(points.shape[0])
    plt.subplot(2, 2, num_plt)
    plt.hist(num_points_list)
    plt.xlabel("{} - {}".format(cls, "number of points"))
    plt.ylabel("frequency")
    num_plt += 1
plt.show()