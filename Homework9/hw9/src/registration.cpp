//
// Created by yu on 11.06.20.
//

#include "registration.hpp"


void readBinaryAndVoxelDown(const std::string& fileName, PointCloud& cloud, NormalCloud& normals, float voxel_size)
{
    PointCloudwithNormal cloud_raw;

    std::fstream input(fileName.c_str(), std::ios::in | std::ios::binary);
    input.seekg(0, std::ios::beg);

    if (not input.good())
    {
        std::cerr << "Read file " << fileName << " failed!";
        exit(EXIT_FAILURE);
    }


    for (int i = 0; input.good() and !input.eof(); i++)
    {
        pcl::PointNormal point_normal_tmp;
        input.read((char*)&point_normal_tmp.x, 3*sizeof(float));
        input.read((char*)&point_normal_tmp.normal_x, 3*sizeof(float));
        cloud_raw.points.emplace_back(point_normal_tmp);
    }
    input.close();

    cloud_raw.height = 1;
    cloud_raw.width = cloud_raw.points.size();
    cloud_raw.is_dense = true;

//    DEBUG(cloud_raw.points.size() << " points read!");

    pcl::VoxelGrid<pcl::PointNormal> voxel_filter;
    voxel_filter.setInputCloud(cloud_raw.makeShared());
    voxel_filter.setDownsampleAllData(true);
    voxel_filter.setLeafSize(voxel_size, voxel_size, voxel_size);
    voxel_filter.filter(cloud_raw);

    size_t filtered_size = cloud_raw.size();

    cloud.points.reserve(filtered_size);
    normals.points.reserve(filtered_size);
    for (size_t i = 0; i < filtered_size; i++)
    {
        pcl::PointXYZ point_tmp;
        pcl::Normal normal_tmp;
        point_tmp.x = cloud_raw.points[i].x;
        point_tmp.y = cloud_raw.points[i].y;
        point_tmp.z = cloud_raw.points[i].z;
        normal_tmp.normal_x = cloud_raw.points[i].normal_x;
        normal_tmp.normal_y = cloud_raw.points[i].normal_y;
        normal_tmp.normal_z = cloud_raw.points[i].normal_z;
        cloud.points.emplace_back(point_tmp);
        normals.points.emplace_back(normal_tmp);
    }

    cloud.height = 1;
    cloud.width = cloud.size();
    cloud.is_dense = true;
    normals.height = 1;
    normals.width = normals.size();
    normals.is_dense = true;
    DEBUG("Filtered cloud size " << cloud.size());
}


void readBinaryAndNormalSpaceDown(const std::string& fileName, PointCloud& cloud, NormalCloud& normals)
{
    PointCloudwithNormal cloud_raw;

    std::fstream input(fileName.c_str(), std::ios::in | std::ios::binary);
    input.seekg(0, std::ios::beg);

    if (not input.good())
    {
        std::cerr << "Read file " << fileName << "failed!";
        exit(EXIT_FAILURE);
    }

//    for (int i = 0; input.good() and !input.eof(); i++)
//    for (int i = 0; i < 20000; i++)
//    {
//        pcl::PointXYZ point;
//        pcl::Normal normal;
//        input.read((char*)&point.x, 3*sizeof(float));
//        input.read((char*)&normal.normal_x, 3*sizeof(float));
//        cloud.points.emplace_back(point);
//        normals.points.emplace_back(normal);
//    }

// TODO
    for (int i = 0; input.good() and !input.eof(); i++)
    {
        pcl::PointNormal point_normal_tmp;
        input.read((char*)&point_normal_tmp.x, 3*sizeof(float));
        input.read((char*)&point_normal_tmp.normal_x, 3*sizeof(float));
        cloud_raw.points.emplace_back(point_normal_tmp);
    }
    input.close();

    cloud_raw.height = 1;
    cloud_raw.width = cloud_raw.points.size();
    cloud_raw.is_dense = true;

    DEBUG(cloud_raw.points.size() << " points read!");

    pcl::NormalSpaceSampling<pcl::PointNormal, pcl::PointNormal> normal_space_down;
    normal_space_down.setInputCloud(cloud_raw.makeShared());
    normal_space_down.setNormals(cloud_raw.makeShared());
    //TODO
    normal_space_down.setBins (12, 12, 12);
    normal_space_down.setSeed (0);
    normal_space_down.setSample (cloud_raw.points.size() / 4);
    normal_space_down.filter(cloud_raw);


    size_t filtered_size = cloud_raw.size();

//    cloud.points.resize(filtered_size);
//    normals.points.resize(filtered_size);
//    for (size_t i = 0; i < filtered_size; i++)
//    {
//        cloud.points[i].x = cloud_raw.points[i].x;
//        cloud.points[i].y = cloud_raw.points[i].y;
//        cloud.points[i].z = cloud_raw.points[i].z;
//        normals.points[i].normal_x = cloud_raw.points[i].normal_x;
//        normals.points[i].normal_y = cloud_raw.points[i].normal_y;
//        normals.points[i].normal_z = cloud_raw.points[i].normal_z;
//    }

    cloud.points.reserve(filtered_size);
    normals.points.reserve(filtered_size);
    for (size_t i = 0; i < filtered_size; i++)
    {
        pcl::PointXYZ point_tmp;
        pcl::Normal normal_tmp;
        point_tmp.x = cloud_raw.points[i].x;
        point_tmp.y = cloud_raw.points[i].y;
        point_tmp.z = cloud_raw.points[i].z;
        normal_tmp.normal_x = cloud_raw.points[i].normal_x;
        normal_tmp.normal_y = cloud_raw.points[i].normal_y;
        normal_tmp.normal_z = cloud_raw.points[i].normal_z;
        normal_tmp.data_c[0] = 0;
        normal_tmp.data_c[1] = 0;
        normal_tmp.data_c[2] = 0;
        normal_tmp.data_c[3] = 0;
        cloud.points.emplace_back(point_tmp);
        normals.points.emplace_back(normal_tmp);
    }

    cloud.height = 1;
    cloud.width = cloud.size();
    cloud.is_dense = true;
    normals.height = 1;
    normals.width = normals.size();
    normals.is_dense = true;
    DEBUG("Filtered cloud size " << cloud.size());
}


void transformCloudInplace(PointCloud& cloud,
                           const Eigen::Matrix3f& R,
                           const Eigen::Vector3f& t)
{
    Eigen::Vector3f point;
    for (size_t i = 0; i < cloud.size(); i++)
    {
        point << cloud.points[i].x, cloud.points[i].y, cloud.points[i].z;
        point = R * point + t;
        cloud.points[i].x = point(0);
        cloud.points[i].y = point(1);
        cloud.points[i].z = point(2);
    }
}


void transformNormalsInplace(NormalCloud& cloud,
                             const Eigen::Matrix3f& R,
                             const Eigen::Vector3f& t)
{
    Eigen::Vector3f normal;
    for (size_t i = 0; i < cloud.size(); i++)
    {
        normal << cloud.points[i].normal_x, cloud.points[i].normal_y, cloud.points[i].normal_z;
        normal = R * normal;
        cloud.points[i].normal_x = normal(0);
        cloud.points[i].normal_y = normal(1);
        cloud.points[i].normal_z = normal(2);
    }
}


void Registration::getISSKeypoints(const PointCloud& input_cloud,
                                   PointCloud& keypoints_cloud,
                                   pcl::PointIndicesConstPtr& keypoints_indices)
{
    pcl::search::KdTree<PointT>::Ptr kdtree (new pcl::search::KdTree<PointT>());
    pcl::ISSKeypoint3D<PointT, PointT> iss_detector;

    iss_detector.setSearchMethod (kdtree);
    iss_detector.setSalientRadius (m_iss_salient_radius);
    iss_detector.setNonMaxRadius (m_iss_non_max_radius);
    iss_detector.setThreshold21 (m_iss_gamma_21);
    iss_detector.setThreshold32 (m_iss_gamma_32);
    iss_detector.setMinNeighbors (m_iss_min_neighbors);
    iss_detector.setNumberOfThreads (m_iss_threads);
    iss_detector.setInputCloud (input_cloud.makeShared());
    iss_detector.compute(keypoints_cloud);
    keypoints_cloud.is_dense = true;

//    DEBUG("keypoints_cloud size " << keypoints_cloud.size());

    keypoints_indices = iss_detector.getKeypointsIndices();
}


void Registration::getHarris3DKeypoints(const PointCloud& input_cloud,
                                        const NormalCloud& input_normals,
                                        PointCloud& keypoints_cloud)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr p_keypoints (new pcl::PointCloud<pcl::PointXYZI> ());
    pcl::HarrisKeypoint3D<pcl::PointXYZ, pcl::PointXYZI> harris_detector;

    harris_detector.setNormals(input_normals.makeShared());

    harris_detector.setNonMaxSupression (m_harris3d_is_nms);
    harris_detector.setRadius(m_harris3d_radius);
    harris_detector.setRefine(m_harris3d_is_refine);
    harris_detector.setThreshold(m_harris3d_nms_threshold);

    harris_detector.setNumberOfThreads(m_harris3d_threads);
    harris_detector.setInputCloud(input_cloud.makeShared());
    harris_detector.compute(*p_keypoints);

    keypoints_cloud.points.resize(p_keypoints->size());
    for (size_t i = 0; i < p_keypoints->size(); i++)
    {
        keypoints_cloud.points[i].x = p_keypoints->points[i].x;
        keypoints_cloud.points[i].y = p_keypoints->points[i].y;
        keypoints_cloud.points[i].z = p_keypoints->points[i].z;
    }

    keypoints_cloud.height = 1;
    keypoints_cloud.width = keypoints_cloud.size();
    keypoints_cloud.is_dense = true;
}



void Registration::getFPFH33Descriptors(const PointCloud& input_cloud,
                                        const PointCloud& input_keypoints_cloud,
                                        const NormalCloud& input_normals,
                                        pcl::PointCloud<pcl::FPFHSignature33>& fpfh_descriptors)
{
    pcl::FPFHEstimationOMP<PointT, NormalT, pcl::FPFHSignature33> fpfh;
    fpfh.setInputCloud(input_keypoints_cloud.makeShared());
    fpfh.setInputNormals(input_normals.makeShared());
    fpfh.setSearchSurface(input_cloud.makeShared());

    pcl::search::KdTree<PointT>::Ptr kdtree (new pcl::search::KdTree<PointT>());
    fpfh.setSearchMethod(kdtree);

    fpfh.setRadiusSearch(m_fpfh_feature_radius);
    fpfh.compute(fpfh_descriptors);
}


void Registration::getSHOT352Descirptors(const PointCloud& input_cloud,
                                         const PointCloud& input_keypoints_cloud,
                                         const NormalCloud& input_normals,
                                         pcl::PointCloud<pcl::SHOT352>& shot_descriptors)
{
    pcl::SHOTEstimationOMP<PointT, NormalT, pcl::SHOT352> descr_est;
    descr_est.setRadiusSearch (m_shot_feature_radius);

    descr_est.setInputCloud (input_keypoints_cloud.makeShared());
    descr_est.setInputNormals (input_normals.makeShared());
    descr_est.setSearchSurface (input_cloud.makeShared());

    descr_est.compute (shot_descriptors);
}


void Registration::RANSAC(const std::vector<std::vector<size_t>>& correspondences,
//                          const NormalCloud& extracted_normals_source,
                          const PointCloud& kp_cloud_source,
//                          const NormalCloud& extracted_normals_target,
                          const PointCloud& kp_cloud_target,
                          Eigen::Matrix3f& R,
                          Eigen::Vector3f& t)
{
    size_t max_consensus_set_size = 0;

    std::random_device rd;
    std::mt19937 mt(rd());

    std::uniform_int_distribution<size_t> dist(0, correspondences.size() - 1);
//    ///
//    // for correspondencesInter
//    std::vector<float> probability(correspondences.size());
//    for (size_t i = 0; i < correspondences.size(); i++)
//    {
//        probability[i] = static_cast<float>(correspondences.size() - (i + 1));
//    }
//    std::discrete_distribution<size_t> dist(probability.begin(), probability.end());
//    ///
    Eigen::Matrix<float, 3, 4> source_mat, target_mat; // 3 x N, N point number
    Eigen::Matrix3f U, V, R_;
    Eigen::Vector3f source_center, target_center, t_;
    Eigen::Vector3f source_point, target_point, source_normal, target_normal;

    for (size_t iter = 0; iter < m_RANSAC_max_iter; iter++)
    {
        size_t random_idx[4];
        while(true)
        {
            // pick randomly 4 correspondences
            random_idx[0] = dist(mt);
            for (size_t i = 1; i < 4; i++)
            {
                random_idx[i] = dist(mt);
                for (size_t j = 0; j < i; j++)
                {
                    while (random_idx[i] == random_idx[j])
                        random_idx[i] = dist(mt);
                }
            }

            // check coplane on selected source keypoints
            float p0[3] = {kp_cloud_source.points[correspondences[random_idx[0]][0]].x,
                           kp_cloud_source.points[correspondences[random_idx[0]][0]].y,
                           kp_cloud_source.points[correspondences[random_idx[0]][0]].z};

            float p1[3] = {kp_cloud_source.points[correspondences[random_idx[1]][0]].x - p0[0],
                           kp_cloud_source.points[correspondences[random_idx[1]][0]].y - p0[1],
                           kp_cloud_source.points[correspondences[random_idx[1]][0]].z - p0[2]};

            float p2[3] = {kp_cloud_source.points[correspondences[random_idx[2]][0]].x - p0[0],
                           kp_cloud_source.points[correspondences[random_idx[2]][0]].y - p0[1],
                           kp_cloud_source.points[correspondences[random_idx[2]][0]].z - p0[2]};

            float p3[3] = {kp_cloud_source.points[correspondences[random_idx[3]][0]].x - p0[0],
                           kp_cloud_source.points[correspondences[random_idx[3]][0]].y - p0[1],
                           kp_cloud_source.points[correspondences[random_idx[3]][0]].z - p0[2]};

            float normal[3] = {(p1[1] * p2[2] - p1[2] * p2[1]),
                               (p1[2] * p2[0] - p1[0] * p2[2]),
                               (p1[0] * p2[1] - p1[1] * p2[0])};

            float normal_length = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
            float distance = (normal[0] * p3[0] + normal[1] * p3[1] + normal[2] * p3[2]) / normal_length;
            if (distance > 0.15)
                break; // 4 points are not coplane
        }

        for (size_t i = 0; i < 4; i++)
        {
            size_t source_idx = correspondences[random_idx[i]][0];
            size_t target_idx = correspondences[random_idx[i]][1];
            source_mat.block(0, i, 3, 1) << kp_cloud_source.points[source_idx].x,
                kp_cloud_source.points[source_idx].y,
                kp_cloud_source.points[source_idx].z;
            target_mat.block(0, i, 3, 1) << kp_cloud_target.points[target_idx].x,
                kp_cloud_target.points[target_idx].y,
                kp_cloud_target.points[target_idx].z;
        }

        // Normalization
        source_center = source_mat.rowwise().mean();
        target_center = target_mat.rowwise().mean();
        source_mat = source_mat.colwise() - source_center;
        target_mat = target_mat.colwise() - target_center;

        // SVD
        Eigen::JacobiSVD<Eigen::Matrix3f> svd(target_mat * source_mat.transpose(), Eigen::ComputeFullU | Eigen::ComputeFullV);
        U = svd.matrixU();
        V = svd.matrixV();
        R_ = U * V.transpose();
        t_ = target_center - R_ * source_center;

        // make R_ always has det = 1
        float R_det = R_.determinant();
        if (R_det < 0)
        {
            Eigen::Matrix3f B = Eigen::Matrix3f::Identity(3, 3);
            B(2, 2) = R_det;
            R_ = V * B * U.transpose();
        }

        // get every consensus set size
        size_t consensus_set_size = 0;
        for (const auto& idx_pair : correspondences)
        {
            // check distance between source_idx and target_idx are small
            source_point << kp_cloud_source.points[idx_pair[0]].x,
                            kp_cloud_source.points[idx_pair[0]].y,
                            kp_cloud_source.points[idx_pair[0]].z;
            target_point << kp_cloud_target.points[idx_pair[1]].x,
                            kp_cloud_target.points[idx_pair[1]].y,
                            kp_cloud_target.points[idx_pair[1]].z;
            float distance = (target_point - (R_ * source_point + t_)).norm();

//            // check normals at source_idx and target_idx are similar
//            source_normal << extracted_normals_source.points[idx_pair[0]].normal_x,
//                extracted_normals_source.points[idx_pair[0]].normal_y,
//                extracted_normals_source.points[idx_pair[0]].normal_z;
//            target_normal << extracted_normals_target.points[idx_pair[1]].normal_x,
//                extracted_normals_target.points[idx_pair[1]].normal_y,
//                extracted_normals_target.points[idx_pair[1]].normal_z;
//            source_normal = R_ * source_normal + t_;
//
//            float product = target_normal.dot(source_normal) / source_normal.norm() / target_normal.norm();
//            float angle = std::acos(product) * 180.0 / 3.14159265;

            if ( distance <= m_RANSAC_dist_threshold/* and angle < m_RANSAC_angle_threshold*/)
                consensus_set_size++;
        }

//        DEBUG("consensus_set_size = " << consensus_set_size);

        if (consensus_set_size > max_consensus_set_size)
        {
            max_consensus_set_size = consensus_set_size;
            R = R_;
            t = t_;
        }
    }

    DEBUG("max_consensus_set_size = " << max_consensus_set_size);
}


void Registration::findRANSACCorrespondencesInter(const pcl::PointCloud<pcl::FPFHSignature33>& fpfh_source,
                                                  const pcl::PointCloud<pcl::FPFHSignature33>& fpfh_target,
                                                  std::vector<std::vector<size_t>>& correspondences)
{

    // convert descriptors to eigen matrix
    size_t N_source = fpfh_source.points.size();
    size_t N_target = fpfh_target.points.size();
    Eigen::MatrixXf fpfh_source_mat = fpfh_source.getMatrixXfMap().transpose();
    Eigen::MatrixXf fpfh_target_mat = fpfh_target.getMatrixXfMap().transpose();

    assert(fpfh_source_mat.cols() == 33);
    assert(fpfh_target_mat.cols() == 33);

    // establish correspondences
    std::vector<size_t> idxpair_source2target(N_source, std::numeric_limits<size_t>::max());
    std::vector<size_t> idxpair_target2source(N_target, std::numeric_limits<size_t>::max());
    std::vector<float>  dist_source2target(N_source, -1.0f);

    // search every target point to find nearest source point
    typedef nanoflann::KDTreeEigenMatrixAdaptor<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> my_kd_tree_t;
    my_kd_tree_t fpfh_source_mat_index(33, std::cref(fpfh_source_mat), 2 /* max leaf */);
    fpfh_source_mat_index.index->buildIndex();
    // 1nn search
    for (size_t i = 0; i < N_target; i++)
    {
        // TODO make result_set definition out of this for loop
        std::vector<size_t> result_indices(1);
        std::vector<float> result_dists(1);
        nanoflann::KNNResultSet<float> result_set(1);
        result_set.init(&result_indices[0], &result_dists[0]);

        const float* query_ptr = fpfh_target.points[i].histogram;
        std::vector<float> query {query_ptr, query_ptr + 33};
        fpfh_source_mat_index.index->findNeighbors(result_set, &query[0], nanoflann::SearchParams(10));

        idxpair_target2source[i] = result_indices[0];
    }

    // search every source point to find nearest target point
    my_kd_tree_t fpfh_target_mat_index(33, std::cref(fpfh_target_mat), 2 /* max leaf */);
    fpfh_target_mat_index.index->buildIndex();
    // 1nn search
    for (size_t i = 0; i < N_source; i++)
    {
        // TODO make result_set definition out of this for loop
        std::vector<size_t> result_indices(1);
        std::vector<float> result_dists(1);
        nanoflann::KNNResultSet<float> result_set(1);
        result_set.init(&result_indices[0], &result_dists[0]);

        const float* query_ptr = fpfh_source.points[i].histogram;
        std::vector<float> query {query_ptr, query_ptr + 33};
        fpfh_target_mat_index.index->findNeighbors(result_set, &query[0], nanoflann::SearchParams(10));

        idxpair_source2target[i] = result_indices[0];
        dist_source2target[i] = result_dists[0];
    }


    for (size_t source_idx = 0; source_idx < N_source; source_idx++)
    {
        size_t target_idx = idxpair_source2target[source_idx];

        if (target_idx == std::numeric_limits<size_t>::max())
            continue;

        if (idxpair_target2source[target_idx] == source_idx)
        {
            correspondences.emplace_back(std::vector<size_t>{source_idx, target_idx});
        }
    }

    // remove worst x% of correspondences
    std::vector<float> correspondences_dist(correspondences.size());
    std::vector<size_t> index(correspondences.size());
    for (size_t i = 0; i < correspondences.size(); i++)
    {
        correspondences_dist[i] = dist_source2target[correspondences[i][0]];
        index[i] = i;
    }

    std::sort(index.begin(), index.end(), [&](const size_t& a, const size_t& b){
        return correspondences_dist[a] < correspondences_dist[b];
    });

    size_t N_inliers = std::floor((1 - m_RANSAC_corres_rejection_rate) * correspondences.size());
    std::vector<std::vector<size_t>> temp(N_inliers, std::vector<size_t>(2));

    for (size_t i = 0; i < N_inliers; i++)
    {
        temp[i] = correspondences[index[i]];
    }
    correspondences.clear();
    correspondences = temp;
}


void Registration::findRANSACCorrespondencesUnion(const pcl::PointCloud<pcl::FPFHSignature33>& fpfh_source,
                                            const pcl::PointCloud<pcl::FPFHSignature33>& fpfh_target,
                                            std::vector<std::vector<size_t>>& correspondences)
{
    struct idx_dist_pair
    {
        size_t idx_src;
        size_t idx_tar;
        float dist;
    };

    // convert descriptors to eigen matrix
    size_t N_source = fpfh_source.points.size();
    size_t N_target = fpfh_target.points.size();
    Eigen::MatrixXf fpfh_source_mat = fpfh_source.getMatrixXfMap().transpose();
    Eigen::MatrixXf fpfh_target_mat = fpfh_target.getMatrixXfMap().transpose();

    std::vector<idx_dist_pair> correspondences_temp;
    correspondences_temp.reserve(N_source + N_target);


    assert(fpfh_source_mat.cols() == 33);
    assert(fpfh_target_mat.cols() == 33);

    // establish correspondences
    // search every target point to find nearest source point
    typedef nanoflann::KDTreeEigenMatrixAdaptor<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> my_kd_tree_t;
    my_kd_tree_t fpfh_source_mat_index(33, std::cref(fpfh_source_mat), 2 /* max leaf */);
    fpfh_source_mat_index.index->buildIndex();
    // 1nn search
    for (size_t i = 0; i < N_target; i++)
    {
        std::vector<size_t> result_indices(1);
        std::vector<float> result_dists(1);
        nanoflann::KNNResultSet<float> result_set(1);
        result_set.init(&result_indices[0], &result_dists[0]);

        const float* query_ptr = fpfh_target.points[i].histogram;
        std::vector<float> query {query_ptr, query_ptr + 33};
        fpfh_source_mat_index.index->findNeighbors(result_set, &query[0], nanoflann::SearchParams(10));

        correspondences_temp.emplace_back(idx_dist_pair{result_indices[0], i, result_dists[0]});
    }

    // search every source point to find nearest target point
    my_kd_tree_t fpfh_target_mat_index(33, std::cref(fpfh_target_mat), 2 /* max leaf */);
    fpfh_target_mat_index.index->buildIndex();
    // 1nn search
    for (size_t i = 0; i < N_source; i++)
    {
        std::vector<size_t> result_indices(1);
        std::vector<float> result_dists(1);
        nanoflann::KNNResultSet<float> result_set(1);
        result_set.init(&result_indices[0], &result_dists[0]);

        const float* query_ptr = fpfh_source.points[i].histogram;
        std::vector<float> query {query_ptr, query_ptr + 33};
        fpfh_target_mat_index.index->findNeighbors(result_set, &query[0], nanoflann::SearchParams(10));

        correspondences_temp.emplace_back(idx_dist_pair{i, result_indices[0],result_dists[0]});
    }


    // remove worst x% of correspondences
    std::sort(correspondences_temp.begin(),
              correspondences_temp.end(),
              [](const auto& a, const auto& b){
        return a.dist < b.dist;
    });

    size_t N_inliers = std::floor((1 - m_RANSAC_corres_rejection_rate) * correspondences_temp.size());
    correspondences.resize(N_inliers);

    for (size_t i = 0; i < N_inliers; i++)
    {
        correspondences[i] = std::vector<size_t> {
            correspondences_temp[i].idx_src,
            correspondences_temp[i].idx_tar,
        };
    }
}



void Registration::getKeypointsNormals(const NormalCloud& input_normals,
                                       const pcl::PointIndicesConstPtr& keypoints_indices,
                                       NormalCloud& extracted_normals)
{
    pcl::ExtractIndices<NormalT> filter;
    filter.setInputCloud(input_normals.makeShared());
    filter.setIndices(keypoints_indices);
    filter.filter(extracted_normals);
}


void Registration::normalSpaceSampling(const PointCloud& input_cloud,
                                       const NormalCloud& input_normals,
                                       PointCloud& sampled_cloud,
                                       NormalCloud& sampled_normals)
{
    pcl::NormalSpaceSampling<PointT, NormalT> normal_space_down;
    normal_space_down.setInputCloud(input_cloud.makeShared());
    normal_space_down.setNormals(input_normals.makeShared());
    normal_space_down.setBins (m_ICP_normal_bins, m_ICP_normal_bins, m_ICP_normal_bins);
    normal_space_down.setSeed (0);
    normal_space_down.setSample (m_ICP_sampled_size);
    normal_space_down.filter(sampled_cloud);

    std::vector<int> indices;
    normal_space_down.filter(indices);

    assert(indices.size() == m_ICP_sampled_size);
    assert(sampled_cloud.size() == m_ICP_sampled_size);

    NormalCloud new_normals;
    new_normals.points.resize(m_ICP_sampled_size);
    new_normals.height = 1;
    new_normals.width = m_ICP_sampled_size;
    new_normals.is_dense = true;
    for (size_t i = 0; i < m_ICP_sampled_size; i++)
    {
        new_normals.points[i].normal_x = input_normals.points[indices[i]].normal_x;
        new_normals.points[i].normal_y = input_normals.points[indices[i]].normal_y;
        new_normals.points[i].normal_z = input_normals.points[indices[i]].normal_z;
    }
    sampled_normals = new_normals;
}



void Registration::VoxelGridSampling(const PointCloud& input_cloud,
                                     const NormalCloud& input_normals,
                                     PointCloud& sampled_cloud,
                                     NormalCloud& sampled_normals)
{
    pcl::PointCloud<pcl::PointNormal> cloud_with_normals;
    cloud_with_normals.height = 1;
    cloud_with_normals.width = input_cloud.size();
    cloud_with_normals.is_dense = true;
    cloud_with_normals.points.resize(input_cloud.size());
    for (size_t i = 0; i < input_cloud.size(); i++)
    {
        cloud_with_normals.points[i].x = input_cloud.points[i].x;
        cloud_with_normals.points[i].y = input_cloud.points[i].y;
        cloud_with_normals.points[i].z = input_cloud.points[i].z;
        cloud_with_normals.points[i].normal_x = input_normals.points[i].normal_x;
        cloud_with_normals.points[i].normal_y = input_normals.points[i].normal_y;
        cloud_with_normals.points[i].normal_z = input_normals.points[i].normal_z;
    }

    pcl::VoxelGrid<pcl::PointNormal> voxel_filter;
    voxel_filter.setInputCloud(cloud_with_normals.makeShared());
    voxel_filter.setLeafSize(1.75, 1.75, 1.75);
    voxel_filter.filter(cloud_with_normals);

    sampled_cloud.height = 1;
    sampled_cloud.width = cloud_with_normals.size();
    sampled_cloud.is_dense = true;
    sampled_cloud.points.resize(cloud_with_normals.size());
    sampled_normals.height = 1;
    sampled_normals.width = cloud_with_normals.size();
    sampled_normals.is_dense = true;
    sampled_normals.points.resize(cloud_with_normals.size());
    for (size_t i = 0; i < cloud_with_normals.size(); i++)
    {
        sampled_cloud.points[i].x = cloud_with_normals.points[i].x;
        sampled_cloud.points[i].y = cloud_with_normals.points[i].y;
        sampled_cloud.points[i].z = cloud_with_normals.points[i].z;
        sampled_normals.points[i].normal_x = cloud_with_normals.points[i].normal_x;
        sampled_normals.points[i].normal_y = cloud_with_normals.points[i].normal_y;
        sampled_normals.points[i].normal_z = cloud_with_normals.points[i].normal_z;
    }
}


void Registration::ICPpoint2plane(const Eigen::Matrix3f& init_R,
                       const Eigen::Vector3f& init_t,
                       const PointCloud& input_cloud_src,
                       const PointCloud& input_cloud_tar,
                       const NormalCloud& input_normals_src,
                       const NormalCloud& input_normals_tar,
                       Eigen::Matrix3f& R,
                       Eigen::Vector3f& t)
{
    // apply init transformation on source points and normals
    PointCloud trans_cloud_src = input_cloud_src;
    NormalCloud trans_normals_src = input_normals_src;
    transformCloudInplace(trans_cloud_src, R, t);
    transformNormalsInplace(trans_normals_src, R, t);

    // normal space downsampling
    PointCloud sampled_cloud_src, sampled_cloud_tar;
    NormalCloud sampled_normals_src, sampled_normals_tar;
    normalSpaceSampling(trans_cloud_src, trans_normals_src, sampled_cloud_src, sampled_normals_src);
    normalSpaceSampling(input_cloud_tar, input_normals_tar, sampled_cloud_tar, sampled_normals_tar);

//    sampled_cloud_src = trans_cloud_src; sampled_normals_src = trans_normals_src;
//    sampled_cloud_tar = input_cloud_tar; sampled_normals_tar = input_normals_tar;

//    VoxelGridSampling(trans_cloud_src, trans_normals_src, sampled_cloud_src, sampled_normals_src);
//    VoxelGridSampling(input_cloud_tar, input_normals_tar, sampled_cloud_tar, sampled_normals_tar);

//    /// check the normal space downsampling ///
//    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
//    viewer->addPointCloud<PointT>(sampled_cloud_src.makeShared(), "source");
//    viewer->addPointCloud<PointT>(sampled_cloud_tar.makeShared(), "target");
//    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0., 0., 1., "source");
//    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1., 0., 1., "target");
//    viewer->setBackgroundColor(0, 0, 0);
//    while (!viewer->wasStopped())
//    {
//        viewer->spinOnce(100);
//    }
//    ///

    // build kdtree for target cloud
    Eigen::MatrixXf tar_mat = sampled_cloud_tar.getMatrixXfMap().block(0, 0, 3, sampled_cloud_tar.size()).transpose();
    typedef nanoflann::KDTreeEigenMatrixAdaptor<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> my_kd_tree_t;
    my_kd_tree_t tar_mat_index(3, std::cref(tar_mat), 2 /* max leaf */);
    tar_mat_index.index->buildIndex();

    Eigen::Matrix3f R_delta;
    Eigen::Vector3f t_delta;
    Eigen::Matrix4f T_delta = Eigen::MatrixXf::Identity(4, 4);
    Eigen::Matrix4f T_total = Eigen::MatrixXf::Identity(4, 4);
    T_total.block(0, 0, 3, 3) = init_R;
    T_total.block(0, 3, 3, 1) = init_t;

    float   last_loss = 0;
    size_t  unchanged_count = 0;
    for (size_t iter = 0; iter < m_ICP_max_iter; iter++)
    {
        // build correspondences
        std::vector<std::vector<size_t>> ICP_corres;
        size_t N_src = sampled_cloud_src.size();

        // 1nn search
        for (size_t i = 0; i < N_src; i++)
        {
            std::vector<size_t> result_indices(1);
            std::vector<float> result_dists(1);
            nanoflann::KNNResultSet<float> result_set(1);
            result_set.init(&result_indices[0], &result_dists[0]);

            float query[3] = {sampled_cloud_src.points[i].x, sampled_cloud_src.points[i].y, sampled_cloud_src.points[i].z};
            tar_mat_index.index->findNeighbors(result_set, &query[0], nanoflann::SearchParams(10));

            // store valid correspondences
            if (result_dists[0] < m_ICP_max_corres_dist)
                ICP_corres.emplace_back(std::vector<size_t>{i, result_indices[0]});
        }

        // compute loss
        Eigen::MatrixXf A_;
        Eigen::VectorXf b_;
        Eigen::VectorXf x_;
        A_.resize(ICP_corres.size(), 6);
        b_.resize(ICP_corres.size(), 1);
        x_.resize(6, 1);
        for (size_t i = 0; i < ICP_corres.size(); i++)
        {
            size_t idx_src = ICP_corres[i][0];
            size_t idx_tar = ICP_corres[i][1];
            float p[3] = {sampled_cloud_src.points[idx_src].x,
                          sampled_cloud_src.points[idx_src].y,
                          sampled_cloud_src.points[idx_src].z};
            float q[3] = {sampled_cloud_tar.points[idx_tar].x,
                          sampled_cloud_tar.points[idx_tar].y,
                          sampled_cloud_tar.points[idx_tar].z};
            float n[3] = {sampled_normals_tar.points[idx_tar].normal_x,
                          sampled_normals_tar.points[idx_tar].normal_y,
                          sampled_normals_tar.points[idx_tar].normal_z};

            A_.block(i, 0, 1, 6) <<
                n[2] * p[1] - n[1] * p[2],
                n[0] * p[2] - n[2] * p[0],
                n[1] * p[0] - n[0] * p[1],
                n[0], n[1], n[2];

            b_(i) = n[0] * q[0] + n[1] * q[1] + n[2] * q[2] - n[0] * p[0] - n[1] * p[1] - n[2] * p[2];
        }

        // TODO need to check the existence of (ATA).inv
        x_ = (A_.transpose() * A_).inverse() * A_.transpose() * b_;

        float loss = (A_ * x_ - b_).squaredNorm();

        if (iter % 10 == 0)
        {
            DEBUG("Iter " << iter << ": Loss = " << loss);
            DEBUG("ICP_corres size = " << ICP_corres.size());
        }

        if (std::abs(last_loss - loss) < m_ICP_loss_epsilon)
        {
            unchanged_count++;
        }

        // check convergence criterium
        if (unchanged_count > 15)
        {
            DEBUG("ICP point2plane Converged!");
            break ;
        }
        else
        {
            last_loss = loss;

            R_delta << 1, -x_[2], x_[1], x_[2], 1, -x_[0], -x_[1], x_[0], 1;
            t_delta << x_[3], x_[4], x_[5];

            T_delta.block(0, 0, 3, 3) = R_delta;
            T_delta.block(0, 3, 3, 1) = t_delta;

            T_total = T_delta * T_total;

            transformCloudInplace(sampled_cloud_src, R_delta, t_delta);

        }
        if (iter == m_ICP_max_iter - 1) DEBUG("ICP max_iter reached!");
    }
    R = T_total.block(0, 0, 3, 3);
    t = T_total.block(0, 3, 3, 1);
//    DEBUG(T_total);
}


void Registration::ICPpoint2point(const Eigen::Matrix3f& init_R,
                                  const Eigen::Vector3f& init_t,
                                  const PointCloud& input_cloud_src,
                                  const PointCloud& input_cloud_tar,
                                  const NormalCloud& input_normals_src,
                                  const NormalCloud& input_normals_tar,
                                  Eigen::Matrix3f& R,
                                  Eigen::Vector3f& t)
{
    // apply init transformation on source points and normals
    PointCloud trans_cloud_src = input_cloud_src;
    NormalCloud trans_normals_src = input_normals_src;
    transformCloudInplace(trans_cloud_src, R, t);
    transformNormalsInplace(trans_normals_src, R, t);

    // normal space downsampling
    PointCloud sampled_cloud_src, sampled_cloud_tar;
    NormalCloud sampled_normals_src, sampled_normals_tar;
    normalSpaceSampling(trans_cloud_src, trans_normals_src, sampled_cloud_src, sampled_normals_src);
    normalSpaceSampling(input_cloud_tar, input_normals_tar, sampled_cloud_tar, sampled_normals_tar);

//    sampled_cloud_src = trans_cloud_src; sampled_normals_src = trans_normals_src;
//    sampled_cloud_tar = input_cloud_tar; sampled_normals_tar = input_normals_tar;

//    VoxelGridSampling(trans_cloud_src, trans_normals_src, sampled_cloud_src, sampled_normals_src);
//    VoxelGridSampling(input_cloud_tar, input_normals_tar, sampled_cloud_tar, sampled_normals_tar);

//    /// check the normal space downsampling ///
//    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
//    viewer->addPointCloud<PointT>(sampled_cloud_src.makeShared(), "source");
//    viewer->addPointCloud<PointT>(sampled_cloud_tar.makeShared(), "target");
//    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0., 0., 1., "source");
//    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1., 0., 1., "target");
//    viewer->setBackgroundColor(0, 0, 0);
//    while (!viewer->wasStopped())
//    {
//        viewer->spinOnce(100);
//    }
//    ///

    // build kdtree for target cloud
    Eigen::MatrixXf tar_mat = sampled_cloud_tar.getMatrixXfMap().block(0, 0, 3, sampled_cloud_tar.size()).transpose();
    typedef nanoflann::KDTreeEigenMatrixAdaptor<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> my_kd_tree_t;
    my_kd_tree_t tar_mat_index(3, std::cref(tar_mat), 2 /* max leaf */);
    tar_mat_index.index->buildIndex();

    Eigen::Matrix3f R_delta;
    Eigen::Vector3f t_delta;
    Eigen::Matrix4f T_delta = Eigen::MatrixXf::Identity(4, 4);
    Eigen::Matrix4f T_total = Eigen::MatrixXf::Identity(4, 4);
    T_total.block(0, 0, 3, 3) = init_R;
    T_total.block(0, 3, 3, 1) = init_t;

    float   last_loss = 0;
    size_t  unchanged_count = 0;
    for (size_t iter = 0; iter < m_ICP_max_iter; iter++)
    {
        // build correspondences
        std::vector<std::vector<size_t>> ICP_corres;
        size_t N_src = sampled_cloud_src.size();

        float loss = 0;
        // 1nn search
        for (size_t i = 0; i < N_src; i++)
        {
            std::vector<size_t> result_indices(1);
            std::vector<float> result_dists(1);
            nanoflann::KNNResultSet<float> result_set(1);
            result_set.init(&result_indices[0], &result_dists[0]);

            float query[3] = {sampled_cloud_src.points[i].x, sampled_cloud_src.points[i].y, sampled_cloud_src.points[i].z};
            tar_mat_index.index->findNeighbors(result_set, &query[0], nanoflann::SearchParams(10));

            // store valid correspondences
            if (result_dists[0] < m_ICP_max_corres_dist)
            {
                ICP_corres.emplace_back(std::vector<size_t>{i, result_indices[0]});
                loss = result_dists[0] * result_dists[0];
            }
        }

        if (iter % 5 == 0)
        {
            DEBUG("Iter " << iter << ": loss = " << loss << ", corres size = " << ICP_corres.size());
        }

        if (std::abs(last_loss - loss) < m_ICP_loss_epsilon)
        {
            unchanged_count++;
        }

        // check convergence criterium
        if (unchanged_count > 15)
        {
            DEBUG("ICP point2point Converged!");
            break ;
        }
        else
        {
            last_loss = loss;

            // compute R_delta t_delta
            Eigen::MatrixXf source_mat, target_mat;
            Eigen::Vector3f source_center, target_center;
            source_mat.resize(3, ICP_corres.size());
            target_mat.resize(3, ICP_corres.size());
            for (size_t i = 0; i < ICP_corres.size(); i++)
            {
                size_t idx_src = ICP_corres[i][0];
                size_t idx_tar = ICP_corres[i][1];
                source_mat.block(0, i, 3, 1) << sampled_cloud_src.points[idx_src].x,
                    sampled_cloud_src.points[idx_src].y,
                    sampled_cloud_src.points[idx_src].z;
                target_mat.block(0, i, 3, 1) << sampled_cloud_tar.points[idx_tar].x,
                    sampled_cloud_tar.points[idx_tar].y,
                    sampled_cloud_tar.points[idx_tar].z;
            }
            source_center = source_mat.rowwise().mean();
            target_center = target_mat.rowwise().mean();

            source_mat = source_mat.colwise() - source_center;
            target_mat = target_mat.colwise() - target_center;
            // SVD
            Eigen::JacobiSVD<Eigen::Matrix3f> svd(target_mat * source_mat.transpose(), Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix3f U = svd.matrixU();
            Eigen::Matrix3f V = svd.matrixV();
            R_delta = U * V.transpose();

            float R_delta_det = R_delta.determinant();
            if (R_delta_det < 0)
            {
                Eigen::Matrix3f B = Eigen::Matrix3f::Identity(3, 3);
                B(2, 2) = R_delta_det;
                R_delta = V * B * U.transpose();
            }

            t_delta = target_center - R_delta * source_center;

            T_delta.block(0, 0, 3, 3) = R_delta;
            T_delta.block(0, 3, 3, 1) = t_delta;
            T_total = T_delta * T_total;
            transformCloudInplace(sampled_cloud_src, R_delta, t_delta);
        }
        if (iter == m_ICP_max_iter - 1) DEBUG("ICP max_iter reached!");
    }

    R = T_total.block(0, 0, 3, 3);
    t = T_total.block(0, 3, 3, 1);
//    DEBUG(T_total);
}


void Registration::compute(const PointCloud& cloud_source,
                           const PointCloud& cloud_target,
                           const NormalCloud& normals_source,
                           const NormalCloud& normals_target,
                           Eigen::Matrix3f& R,
                           Eigen::Vector3f& t)
{
    auto kp_start = Clock::now();
    PointCloud kp_cloud_source;
    PointCloud kp_cloud_target;
    // ISS
    // TODO extracted idx are not used!!!
    pcl::PointIndicesConstPtr kp_idx_source;
    pcl::PointIndicesConstPtr kp_idx_target;
//    getISSKeypoints(cloud_source, kp_cloud_source, kp_idx_source);
//    getISSKeypoints(cloud_target, kp_cloud_target, kp_idx_target);

//    // Harris
    getHarris3DKeypoints(cloud_source, normals_source, kp_cloud_source);
    getHarris3DKeypoints(cloud_target, normals_target, kp_cloud_target);

    auto kp_end = Clock::now();
    Duration kp_time = kp_end - kp_start;

    DEBUG("\nKeypoints detection finished!");
    DEBUG("source keypoint size = " << kp_cloud_source.size());
    DEBUG("target keypoint size = " << kp_cloud_target.size());
    DEBUG("Keypoints detection takes " << kp_time.count() << " ms");

//    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
//    viewer->addPointCloud<PointT>(cloud_source.makeShared(), "cloud_source");
//    viewer->addPointCloud<PointT>(kp_cloud_source.makeShared(), "kp_cloud_source");
//    viewer->addPointCloud<PointT>(cloud_target.makeShared(), "cloud_target");
//    viewer->addPointCloud<PointT>(kp_cloud_target.makeShared(), "kp_cloud_target");
//    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0., 0., 1., "cloud_source");
//    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1., 0., 1., "kp_cloud_source");
//    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "kp_cloud_source");
//    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0., 1., 0., "cloud_target");
//    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1., 0., 0., "kp_cloud_target");
//    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "kp_cloud_target");
//    viewer->setBackgroundColor(0, 0, 0);
//    while (!viewer->wasStopped())
//    {
//        viewer->spinOnce(100);
//    }

//    ///
//    pcl::PointCloud<pcl::SHOT352>:: shot_source (new pcl::PointCloud<pcl::SHOT352> ());
//    pcl::PointCloud<pcl::SHOT352>:: shot_target (new pcl::PointCloud<pcl::SHOT352> ());
//    getSHOT352Descirptors(cloud_source, kp_cloud_source, normals_source, shot_source);
//    getSHOT352Descirptors(cloud_target, kp_cloud_target, normals_target, shot_target);
//    ///
//    DEBUG("shot_source rows = " << shot_source->size());
//    DEBUG("shot_target rows = " << shot_target->size());
//    ///

    auto fpfh_start = Clock::now();
    pcl::PointCloud<pcl::FPFHSignature33> fpfh_source;
    pcl::PointCloud<pcl::FPFHSignature33> fpfh_target;
    getFPFH33Descriptors(cloud_source, kp_cloud_source, normals_source, fpfh_source);
    getFPFH33Descriptors(cloud_target, kp_cloud_target, normals_target, fpfh_target);
    auto fpfh_end = Clock::now();
    Duration fpfh_time = fpfh_end - fpfh_start;
    DEBUG("\nCompute descriptions finished!");
    DEBUG("Compute descriptions takes " << fpfh_time.count() << " ms");

    auto t1 = Clock::now();
    std::vector<std::vector<size_t>> correspondences;
    findRANSACCorrespondencesUnion(fpfh_source, fpfh_target,correspondences);
    auto t2 = Clock::now();
    Duration t_cor1 = t2 - t1;
    DEBUG("\nCompute correspondences finished!");
    DEBUG("correspondences size = " << correspondences.size());
    DEBUG("Compute correspondences takes " << t_cor1.count() << " ms");

    if (correspondences.size() < 4)
    {
        std::cerr << "Correspondences are fewer than 4! Failed!" << std::endl;
        return ;
    }


//    ////
//    PointCloud kpcor_source;
//    PointCloud kpcor_target;
//    for (const auto& idx_pair : correspondences)
//    {
//        kpcor_source.points.emplace_back(kp_cloud_source[idx_pair[0]]);
//        kpcor_target.points.emplace_back(kp_cloud_target[idx_pair[1]]);
//    }
//
//    pcl::visualization::PCLVisualizer::Ptr viewer_corres(new pcl::visualization::PCLVisualizer("3D Viewer"));
//    viewer_corres->setBackgroundColor(0, 0, 0);
//
//    viewer_corres->addPointCloud<PointT>(cloud_source.makeShared(), "cloud_source");
//    viewer_corres->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0., 0., 1., "cloud_source");
//
//    viewer_corres->addPointCloud<PointT>(kpcor_source.makeShared(), "kpcor_source");
//    viewer_corres->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1., 0., 1., "kpcor_source");
//    viewer_corres->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "kpcor_source");
//
//    viewer_corres->addPointCloud<PointT>(cloud_target.makeShared(), "cloud_target");
//    viewer_corres->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0., 1., 0., "cloud_target");
//    viewer_corres->addPointCloud<PointT>(kpcor_target.makeShared(), "kpcor_target");
//
//    viewer_corres->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1., 0., 0., "kpcor_target");
//    viewer_corres->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "kpcor_target");
//    while (!viewer_corres->wasStopped())
//    {
//        viewer_corres->spinOnce(100);
//    }
//    ////

//    NormalCloud extracted_normals_source;
//    NormalCloud extracted_normals_target;
//    getKeypointsNormals(normals_source, kp_idx_source, extracted_normals_source);
//    getKeypointsNormals(normals_target, kp_idx_target, extracted_normals_target);
//    DEBUG("extracted_normals_source size = " << extracted_normals_source.size());
//    DEBUG("extracted_normals_target size = " << extracted_normals_target.size());

    DEBUG("\nRANSAC start!");
    auto ransac_start = Clock::now();
    Eigen::Matrix3f init_R;
    Eigen::Vector3f init_t;
    RANSAC(correspondences, /*extracted_normals_source, */kp_cloud_source, /*extracted_normals_target,*/ kp_cloud_target, init_R, init_t);
    auto ransac_end = Clock::now();
    Duration ransac_time = ransac_end - ransac_start;
    R = init_R;
    t = init_t;
    Eigen::Matrix4f T = Eigen::MatrixXf::Identity(4, 4);
    T.block(0, 0, 3, 3) = R;
    T.block(0, 3, 3, 1) = t;
    DEBUG("RANSAC time " << ransac_time.count() << " ms");
//    DEBUG(T);

    DEBUG("\nICP start!");
    auto icp_start = Clock::now();
    ICPpoint2point(init_R, init_t, cloud_source, cloud_target, normals_source, normals_target, R, t);
    auto icp_end = Clock::now();
    Duration icp_time = icp_end - icp_start;
    DEBUG("ICP time " << icp_time.count() << " ms");
}


