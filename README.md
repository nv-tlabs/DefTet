# DefTet
This is the official repo for:

#### Learning Deformable Tetrahedral Meshes for 3D Reconstruction (NeurIPS 2020)

[Jun Gao](http://www.cs.toronto.edu/~jungao/), [Wenzheng Chen](http://www.cs.toronto.edu/~wenzheng/), [Tommy Xiang](), [Clement Fuji Tsang](), [Alec Jacobson](https://www.cs.toronto.edu/~jacobson/), [Morgan McGuire](https://research.nvidia.com/person/morgan-mcguire), [Sanja Fidler](http://www.cs.toronto.edu/~fidler/)


**NeurIPS 2020, [Paper](https://arxiv.org/abs/2011.01437), [Supplementary](https://nv-tlabs.github.io/DefTet/files/supplement.pdf), [Project Page](https://nv-tlabs.github.io/DefTet/),**

## Requirements
- Python 3.8 is supported.
- Pytorch 1.9.0.
- This code is tested with CUDA 11.1.
- GCC >= 6.0

Install Pytorch
```bash
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

Install QuarTet following [official Link](https://github.com/crawforddoran/quartet).
```bash
git clone https://github.com/crawforddoran/quartet
apt-get install libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev # Dependencies for QuarTet
cd quartet
make depend
make 
```
Install Kaolin following [official Link](https://kaolin.readthedocs.io/en/latest/notes/installation.html).

Install other required libraries by doing:
```bash
apt-get update
apt-get install tmux htop -y
pip install opencv-python tensorboardx meshzoo ipdb imageio
apt-get install ffmpeg libsm6 libxext6  -y
HOME_DIR="$PWD"
cd "$HOME_DIR/utils/lib/tet_adj_share"
bash do_all.sh
cd "$HOME_DIR/utils/lib/tet_face_adj"
bash do_all.sh
cd "$HOME_DIR/utils/lib/tet_point_adj"
bash do_all.sh
cd "$HOME_DIR/utils/lib/colaps_v"
bash do_all.sh
cd $HOME_DIR
```

## Training
### Dataset
We use ShapeNet Core dataset, please first download ShapeNet following [Official Link](https://shapenet.org) 
into the directory  `/data/shapenet_kaolin`, then inside the code, we will use kaolin to preprocess the dataset (see details at `dataloader.py`)

### Training on 3D Point Cloud Reconstruction

We experimented training on different resolution by passing the config `--res xxx`. To reproduce the results in paper, we recommond using res 70, training at higher resolution won't have too much improvement. 
```bash
python train_multigpu.py  --pow 4 --save_vis  --batch_size 8 --print_every 300  --dataset_dir /data/shapenet_kaolin  --save_vis_ever 10000  --no_use_pos_encoding --no_use_vert_feat --use_init_pos_mask --point_cloud --lambda_surf 5 --lambda_surf_chamfer 1 --lambda_amips 1 --res 70 --no_expand_boundary --use_two_encoder --no_use_vert_feat --no_use_pvcnn_pos_decoder --no_use_dvr_pos_decoder --use_gcn_pos_decoder --no_use_dvr_occ_decoder --add_input_noise --use_pvcnn_occ_decoder  --use_all --experiment_id pc_pvcnn_gcn_70_surf_1_chamfer_1_all_scale_pvcnn --scale_pvcnn
```



### Inference on 3D Point Cloud Reconstruction using trained model

This evaluation scripts provides the results on different metrics, including chamfer, chamfer L1, F-score, Hausdorff distance.
```bash
python eval.py ----experiment_path experiments/YOUR-EXP-NAME
```

### Optimization that uses 2D supervision

Here we provide the optimization script on using our diff render for dmtet to optimize a tet mesh using volume rendering. 

```bash
cd diff_render/diftet_6_subdiv/6_optim
pip install configargparse
python optim_with_mask_subdiv_from_gridmov.py --expname hotdog --datadir YOUR_NERF_DIR --savedir YOUR_SAVING_DIR --remote
```


## Ciatation
If you use the code, please cite our paper:
```latex
@inproceedings{gao2020deftet,
title={Learning Deformable Tetrahedral Meshes for 3D Reconstruction},
author={Jun Gao and Wenzheng Chen and Tommy Xiang and Clement Fuji Tsang and Alec Jacobson and Morgan McGuire and Sanja Fidler},
booktitle={Advances In Neural Information Processing Systems},
year={2020}
}
```

## Following works
We have some following works related to this one:
- Deep Marching Tetrahedra: a Hybrid Representation for High-Resolution 3D Shape Synthesis. [NeurIPS 2021](https://nv-tlabs.github.io/DMTet/)
- Extracting Triangular 3D Models, Materials, and Lighting From Images.  [(CVPR) 2022](https://nvlabs.github.io/nvdiffrec/)


##### Acknowledgement
The code for PVCNN module in this codebase is borrowed from [PVCNN](https://github.com/mit-han-lab/pvcnn) (MIT License)
