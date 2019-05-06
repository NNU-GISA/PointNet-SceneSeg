# 3D Semantic Segmentation of virtual kitti dataset using PointNet

This project is a modified version of <a href="https://github.com/kargarisaac/PointNet-SemSeg-VKITTI3D" target="_blank">PointNet-SemSeg-VKITTI3D</a>.

The main code is from <a href="https://github.com/charlesq34/pointnet" target="_blank">PointNet GitHub Repo<a>

## Dataset
You can download the dataset from <a href="https://github.com/VisualComputingInstitute/vkitti3D-dataset" target="_blank">here</a>. 

All files are provided as numpy .npy files. Each file contains a N x F matrix, where N is the number of points in a scene and F is the number of features per point, in this case F=7. The features are XYZRGBL, the 3D XYZ position, the RGB color and the ground truth semantic label L. Each file is for a scene. 

## Training

Once you have downloaded and prepared data, to start training use train_vkitti.py. 

## Visualise data

For data visualization you can use vis_data_vispy.py file.

