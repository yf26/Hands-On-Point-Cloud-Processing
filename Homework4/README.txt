1. ground_detection_ransac.py中实现了基于ransac的地面拟合算法，ransac_on_segments函数和ransac_on_segments_v2函数实现了在不同segments上分段拟合。

2. ground_detection_SVD.py中实现了基于pca法向量估计的地面拟合，并测试了open3d中dbscan的效果。

3. foreground_clustering.py中实现了基于range image BFS的clustering。

4. 测试点云在/test文件夹中，具体结果见/result文件夹。
