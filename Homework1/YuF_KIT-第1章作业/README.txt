1、代码运行结果见homework1.pdf。depth_complement文件夹中完成了额外作业的前两部分，代码运行结果见depth_complement.pdf文件。

2、为了在pca_normal.py中更快的估计法向量，使用了pybind11，用c++写了一个库mylib，其中集成了Open3d中FastEigen3x3函数的源代码（见../my_pybind11/src/mylib.cpp）。大约估算，此方法比用np.linalg.eig做SVD要快一倍。
请在本地pybind11目录下编译/my_pybind11/CMakeLists.txt，并在环境中python setup.py install。注释掉pca_normal.py中的98、99行，去掉96、97行的注释，即可测试该方法。

3、pca_cpp文件夹中用c++实现了点云的主轴分析和法向量估计，对于20000多个点，c++基于eigen做法向量估计比python基于numpy大约快三倍。

4、off_to_ply.py中修改了write_ply_points_only_from_off函数，对ModelNet40中的每个类，只转化了train和test中的一个.ply文件。

5、本作业python代码运行在anaconda建立的环境里，环境配置见environment.yml
