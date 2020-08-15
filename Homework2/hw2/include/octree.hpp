#ifndef HW2_OCTREE_HPP
#define HW2_OCTREE_HPP


#include <vector>
#include <cmath>

double factor[] = {-0.5, 0.5};

class Octant
{
public:
    std::vector<Octant*> children;
    std::vector<double> center;
    double extent;
    std::vector<int> point_indices;
    bool is_leaf;

    Octant(std::vector<Octant*>& ch,
           std::vector<double>& c,
           double& ext,
           std::vector<int>& pt_idxs,
           bool leaf)
    {
        children = ch;
        center = c;
        extent = ext;
        point_indices = pt_idxs;
        is_leaf = leaf;
    }

    static std::vector<Octant*> address_set;

};

Octant* OctreeBuild(Octant*& root,
                    std::vector<std::vector<double>>& db,
                    std::vector<double>& center,
                    double& extent,
                    std::vector<int>& point_indices,
                    int& leaf_size,
                    double& min_extent
                    )
{
    if (point_indices.empty())
        return nullptr;

    if (root == nullptr)
    {
        std::vector<Octant*> empty_children(8, nullptr);
        root = new Octant(empty_children, center, extent, point_indices, true);
        Octant::address_set.emplace_back(root);
    }

    if (point_indices.size() <= leaf_size or extent <= min_extent)
        root->is_leaf = true;
    else
    {
        root->is_leaf = false;
        std::vector<std::vector<int>> children_point_indices(8, std::vector<int>{});
        for (auto& point_idx : point_indices)
        {
            unsigned char morton_code = 0;
            if (db[point_idx][0] > center[0])
                morton_code = morton_code | (unsigned char)1;
            if (db[point_idx][1] > center[1])
                morton_code = morton_code | (unsigned char)2;
            if (db[point_idx][2] > center[2])
                morton_code = morton_code | (unsigned char)4;
            children_point_indices[morton_code].emplace_back(point_idx);
        }

        for (unsigned char i = 0; i < 8; i++)
        {
            double child_center_x, child_center_y, child_center_z, child_extent;
            child_center_x = center[0] + factor[ (i & (unsigned char)1) >0] * extent;
            child_center_y = center[1] + factor[ (i & (unsigned char)2) >0] * extent;
            child_center_z = center[2] + factor[ (i & (unsigned char)4) >0] * extent;
            child_extent = 0.5 * extent;
            std::vector<double> child_center{child_center_x, child_center_y, child_center_z};

            root->children[i] = OctreeBuild(
                root->children[i], db, child_center,
                child_extent, children_point_indices[i], leaf_size, min_extent
            );
        }
    }

    return root;
}


bool inside(std::vector<double>& query, double radius, Octant*& octant)
{
    bool possible_space[3];
    for (int i = 0; i < 3; i++)
        possible_space[i] = (fabs(query[i] - octant->center[i]) + radius) < octant->extent;

    return (possible_space[0] and possible_space[1] and possible_space[2]); // TODO
}


bool overlaps(std::vector<double>& query, double radius, Octant*& octant)
{
    double max_dist = radius + octant->extent;

    double query_offset_abs[3];
    bool outside[3];
    bool contact[3];

    for (int i = 0; i < 3; i++)
    {
        query_offset_abs[i] = fabs(query[i] - octant->center[i]);
        outside[i] = query_offset_abs[i] > max_dist;
        contact[i] = query_offset_abs[i] < octant->extent;
    }

    if(outside[0] or outside[1] or outside[2])
        return false;

    if ( ((int)contact[0] + (int)contact[1] + (int)contact[2]) >= 2 )
        return true;

    double x_diff, y_diff, z_diff;
    x_diff = std::max(query_offset_abs[0] - octant->extent, 0.);
    y_diff = std::max(query_offset_abs[1] - octant->extent, 0.);
    z_diff = std::max(query_offset_abs[2] - octant->extent, 0.);

    return x_diff * x_diff + y_diff * y_diff + z_diff * z_diff < radius * radius;
}


bool contains(std::vector<double>& query, double radius, Octant*& octant)
{
    double max_dist[3];
    for (int i = 0; i < 3; i++)
        max_dist[i] = fabs(query[i] - octant->center[i]) + octant->extent;
    return max_dist[0] * max_dist[0] + max_dist[1] * max_dist[1] + max_dist[2] * max_dist[2] < radius * radius;
}



bool OctreeKNNSearch(Octant*& root, std::vector<std::vector<double>>& db,
                     KNNResultSet& result_set, std::vector<double>& query)
{
    if (root == nullptr)
        return false;

    // traverse the leaf points
    if (root->is_leaf and !root->point_indices.empty())
    {
        for (auto& idx : root->point_indices)
        {
            double diff = sqrt(pow(db[idx][0] - query[0], 2) +
                               pow(db[idx][1] - query[1], 2) +
                               pow(db[idx][2] - query[2], 2));
            result_set.addPoint(diff, idx);
        }
        return inside(query, result_set.getWorstDist(), root);
    }

    // root is not leaf, find the most relevant child octant
    unsigned char morton_code = 0;
    if (query[0] > root->center[0])
        morton_code = morton_code | (unsigned char)1;
    if (query[1] > root->center[1])
        morton_code = morton_code | (unsigned char)2;
    if (query[2] > root->center[2])
        morton_code = morton_code | (unsigned char)4;
    if (OctreeKNNSearch(root->children[morton_code], db, result_set, query))
        return true;

    // check other children
    for (int c = 0; c < 8; c++)
    {
        if (c == morton_code or root->children[c] == nullptr)
            continue;
        if (not overlaps(query, result_set.getWorstDist(), root->children[c]))
            continue;
        if (OctreeKNNSearch(root->children[c], db, result_set, query))
            return true;
    }

    // final check if we can stop search here
    return inside(query, result_set.getWorstDist(), root);
}


bool OctreeRadiusNNSearch(Octant*& root, std::vector<std::vector<double>>& db,
                          RadiusNNResultSet& result_set, std::vector<double>& query)
{
    if (root == nullptr)
        return false;

    // traverse the leaf points
    if ( contains(query, result_set.getWorstDist(), root) )
    {
        for (auto& idx : root->point_indices)
        {
            double diff = sqrt(pow(db[idx][0] - query[0], 2) +
                               pow(db[idx][1] - query[1], 2) +
                               pow(db[idx][2] - query[2], 2));
            result_set.addPoint(diff, idx);
        }
        // don't need to check any child TODO
        return false;
    }

    // traverse the leaf points
    if (root->is_leaf and !root->point_indices.empty())
    {
        for (auto& idx : root->point_indices)
        {
            double diff = sqrt(pow(db[idx][0] - query[0], 2) +
                               pow(db[idx][1] - query[1], 2) +
                               pow(db[idx][2] - query[2], 2));
            result_set.addPoint(diff, idx);
        }
        return inside(query, result_set.getWorstDist(), root);
    }

    // no need to ge to most relevant child first, because anyway we will go through all children

    // check other children
    for (int c = 0; c < 8; c++)
    {
        if (root->children[c] == nullptr)
            continue;
        if (not overlaps(query, result_set.getWorstDist(), root->children[c]))
            continue;
        if (OctreeRadiusNNSearch(root->children[c], db, result_set, query))
            return true;
    }

    // final check if we can stop search here
    return inside(query, result_set.getWorstDist(), root);
}



std::vector<Octant*> Octant::address_set;
Octant* OctreeConstruction(std::vector<std::vector<double>>& db, int leaf_size, double min_extent)
{
    std::vector<int> point_indices(db.size());
    std::iota(point_indices.begin(), point_indices.end(), 0);

    std::vector<double> db_min = db[0];
    std::vector<double> db_max = db[0];
    std::vector<double> db_center = {0, 0, 0};
    double db_extent;

    for (auto& item : db)
    {
        for (int i = 0; i < 3; i++)
        {
            db_min[i] = item[i] < db_min[i]? item[i] : db_min[i];
            db_max[i] = item[i] > db_max[i]? item[i] : db_max[i];
        }
    }

    db_extent = std::max(db_max[0] - db_min[0], std::max(db_max[1] - db_min[1], db_max[2] - db_min[2]));

    for (int i = 0; i < 3; i++)
        db_center[i] = db_min[i] + db_extent;


    Octant* root = nullptr;
    root = OctreeBuild(root, db, db_center, db_extent, point_indices, leaf_size, min_extent);
//    std::cout << Octant::address_set.size() << " Nodes address are saved!" << std::endl;
    return root;
}

void OctreeDestruction()
{
    for (auto& item: Octant::address_set)
        delete item;
}


#endif //HW2_OCTREE_HPP