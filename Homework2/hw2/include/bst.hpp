#ifndef HW2_BST_HPP
#define HW2_BST_HPP

#include <iostream>
#include <cmath>

template <typename ElemType>
class TreeNode
{
public:
    ElemType key;
    TreeNode<ElemType>* left;
    TreeNode<ElemType>* right;
    int value;
};


template<typename ElemType>
void insert(TreeNode<ElemType>*& root, ElemType key, int index)
{
    if (!root)
    {
        root = new TreeNode<ElemType>;
        root->key = key;
        root->left = nullptr;
        root->right = nullptr;
        root->value = index;
    }
    else
    {
        if (key < root->key)
        {
            insert(root->left, key, index);
        }
        else if (key > root->key)
        {
            insert(root->right, key, index);
        }
        else
        {
            std::cout << "TreeNode with key: " << key << " already exists!" << std::endl;
        }
    }
}

template<typename ElemType>
void deleteTree(TreeNode<ElemType>* root) {
    if (root != nullptr)
    {
        deleteTree(root->left);
        deleteTree(root->right);
        delete root;
    }
}

template<typename ElemType>
void inOrder(TreeNode<ElemType>* root) {
    if (root != nullptr)
    {
        inOrder(root->left);
        std::cout << "Key = " << root->key << ", Index = " << root->value << std::endl;
        inOrder(root->right);
    }
}

template<typename ElemType>
void preOrder(TreeNode<ElemType>* root) {
    if (root != nullptr)
    {
        std::cout << root->key << std::endl;
        preOrder(root->left);
        preOrder(root->right);
    }
}

template<typename ElemType>
void postOrder(TreeNode<ElemType>* root) {
    if (root != nullptr)
    {
        postOrder(root->left);
        postOrder(root->right);
        std::cout << root->key << std::endl;
    }
}

template <typename ElemType>
TreeNode<ElemType>* searchRecursively(TreeNode<ElemType>*& root, ElemType queryKey)
{
    if (root == nullptr) return root;
    if (root->key == queryKey)
        return root;
    else
    {
        if (queryKey < root->key)
            return searchRecursively(root->left, queryKey);
        else if (queryKey > root->key)
            return searchRecursively(root->right, queryKey);
    }
}

template <typename ElemType>
TreeNode<ElemType>* searchIteratively(TreeNode<ElemType>*& root, ElemType queryKey)
{
    TreeNode<ElemType>* current = root;
    while (current != nullptr)
    {
        if (current->key == queryKey)
            return current;
        else if (queryKey < current->key)
            current = current->left;
        else if (queryKey > current->key)
            current = current->right;
    }
    return current;
}


template <typename ElemType>
bool KNNSearch(TreeNode<ElemType>*& root, KNNResultSet& resultSet, ElemType queryKey)
{
    if (root == nullptr)
        return false;

    resultSet.addPoint(fabs(root->key - queryKey), root->value);
    if (resultSet.getWorstDist() == 0)
        return true;

    if (queryKey <= root->key)
    {
        if (KNNSearch(root->left, resultSet, queryKey))
            return true;
        else if (fabs(root->key - queryKey) < resultSet.getWorstDist())
            return KNNSearch(root->right, resultSet, queryKey);
        return false;
    }

    if (queryKey > root->key)
    {
        if (KNNSearch(root->right, resultSet, queryKey))
            return true;
        else if (fabs(root->key - queryKey) < resultSet.getWorstDist())
            return KNNSearch(root->left, resultSet, queryKey);
        return false;
    }

}


template <typename ElemType>
bool RadiusNNSearch(TreeNode<ElemType>*& root, RadiusNNResultSet& resultSet, ElemType queryKey)
{
    if (root == nullptr)
        return false;

    resultSet.addPoint(fabs(root->key - queryKey), root->value);

    if (queryKey <= root->key)
    {
        if (RadiusNNSearch(root->left, resultSet, queryKey))
            return true;
        else if (fabs(root->key - queryKey) < resultSet.getWorstDist())
            return RadiusNNSearch(root->right, resultSet, queryKey);
        return false;
    }

    if (queryKey > root->key)
    {
        if (RadiusNNSearch(root->right, resultSet, queryKey))
            return true;
        else if (fabs(root->key - queryKey) < resultSet.getWorstDist())
            return RadiusNNSearch(root->left, resultSet, queryKey);
        return false;
    }

}

#endif //HW2_BST_HPP

