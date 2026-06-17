# GridCellLifting

This project aims to decode trajectories from spike train data of grid cells using topological data analysis (TDA) methods. It includes complete pipeline for decoding the data and implemented validations on both simulated and experimental data. 

Paper: [arXiv](https://arxiv.org/abs/2510.16216)

For a quick demonstration of the method on datasets, run the three notebooks under "example".

## Table of Contents

- [GridCellLifting](#gridcelllifting)
  - [Table of Contents](#table-of-contents)
  - [Setup](#setup)
    - [Environment Installation](#environment-installation)
  - [Components](#components)
    - [Trajectory simulation](#trajectory-simulation)
    - [Grid cell simulation](#grid-cell-simulation)
    - [Persistent homology](#persistent-homology)
    - [Toroidal coordinates](#toroidal-coordinates)
    - [Lifting algorithm](#lifting-algorithm)
    - [Epsilon computation](#epsilon-computation)
    - [Affine transformation \& Mismatch score](#affine-transformation--mismatch-score)
    - [Noise-adding simulation](#noise-adding-simulation)
  - [Authors](#authors)
  - [References](#references)

## Setup

In the development of the repository, `Python 3.11` is used, but in theory any version of `Python 3.8+` should work.

The paths for accessing data and the figures are stored in `constants.py`. Before running the code, please create a file named `DATA_ROOT.py` and define two variables: `DATA_ROOT` (for accessing data and cache files) and `REPO_PATH` (for saving figures). For direct running in the current environment, it can be set as:

```python
DATA_ROOT = '.'
REPO_PATH = '.'
```

### Environment Installation

This project uses Conda for environment management. To install the required dependencies, run:

```bash
conda env create -f requirements.yml
conda activate gridpathlifting
```

If the environment already exists and you need to update it, run:

```bash
conda env update --file requirements.yml
```

## Components

`pipeline.py` is provided for pipeline decoding of the grid cell activity.

### Trajectory simulation

Random walk in an enclosed space is simulated in `trajectory.py`. The world can be configured with size (100*100 by default) and number of holes, inputted with `(LL_x, LL_y, UR_x, UR_y)` coordinates of the holes as a list. The simulated trajactory will be saved as `(trajectory, world)`.

### Grid cell simulation

Grid cell simulation is adopted from *Gardner et al., 2022*. Grid cells are simulated in `grid_cell_simulation.py` using a continuous attractor network (CAN) model. `utils.py` contains helper functions from the original studies.

### Persistent homology

The persistence homology is computed in `persistence_homology.py`. For large data like grid cell spike trian, use `persistence_analysis` that computes with distance matrix. For direct computation of the trajectories, use `trajectory_persistence_analysis`. The persistent homology is computed with `ripser` package.

### Toroidal coordinates

The toroidal coordinates is calculated from the persistence homology results. This part of the project is implemented in [`toroidal_coordinates/`](toroidal_coordinates/) folder.

1. The toroidal coordinates of Gardner et al. are adopted in `Gardner_toroidal.py` with `Gardner_coord`. `compute_coord` computes persistent homology with the most active frames and interpolates the toroidal coordinates for the entire time series.
2. `dreimac_toroidal.py` computes the toroidal coordinates using the DREiMac package. It utilizes the `CircularCoords` function from the package to compute the toroidal coordinates of the grid cell spike train.

### Lifting algorithm

The proposed lifting algorithm decodes trajectories from the toroidal coordinates of the grid cells. The lifting algorithm is implemented in `toroidal_lifting.py`. More details are explained in the paper.

Three versions of the lifting algorithm are implemented:

1. `toroidal_lifting` is a direct implementation of the covering map. It creates covers of the torus and recovers the lifted path by cover transitions. The covers are predefined with the toroidal coordinates.
2. `toroidal_lifting_distance` simplifies the lifting algorithm by finding the lifted path with distance monitoring. An *epsilon* value is used to determine the distance threshold for executing the lifting process.
3. `toroidal_lifting_distance_upgrade` is an upgraded version of the lifting algorithm with distance. It takes extra steps to check the necessity of the lifting and decides whether to lift the path or not. In this version the *epsilon* parameter is used to determine the threshold for checking, and larger *epsilon* values result in better performance slightly increased time.

### Epsilon computation

Our study offers a method to determine the proper *epsilon* value for the lifting algorithm. In `epsilon.py`, the *epsilon* value is computed to include the majority of the points that potentially needs to be lifted. more details are explained in the paper.

### Affine transformation & Reconstruction error

In [`affine_transformation/`](affine_transformation/) folder, `get_transform_mat.py` computes the affine transformation matrix between two paths using `estimateAffine2D` function from the OpenCV package. The affine transformation is used to align the lifted path with the original trajectory.

`score_mismatch.py` computes the reconstruction error between two paths and is used to evaluate the performance of the lifting algorithm. The reconstruction error is defined as the mean Euclidean distance between the two paths, normalized by the size of the world. If not specified, the world size is computed as the maximum horizontal/vertical distance of path2.

### Simulated data

The lifting algorithm is tested on simulated data. See `example/simulated_data.ipynb`

### 1D experimental data

The lifting algorithm is tested on experimental data from 1-dimensional movement from [Wen et al](https://www.nature.com/articles/s41586-024-08034-3). The implementation is done in `example/1D_experimental_data.ipynb.`

Prior to running the notebook, users must do the following:
1. Download the dataset from [Mendeley Data](https://data.mendeley.com/datasets/rgtk6jygjc/1). 
2. Update `WEN_DATA_PATH` within "constants.py".

### 2D experimental data

The lifting algorithm is tested on experimental data from 2-dimensional movement from [Gardner et al](https://www.nature.com/articles/s41586-021-04268-7). The implementation is done in `example/2D_experimental_data.ipynb.`

Prior to running the notebook, users must do the following:
1. Download the dataset from [figshare](https://figshare.com/articles/dataset/Toroidal_topology_of_population_activity_in_grid_cells/16764508?file=35078602)
2. Update `GARDNER_DATA_PATH` within "constants.py"


### Noise in neural activity

To test the robustness of this pipeline, we added noise to the simulated grid cell activity. In `add_noise.py` in [`noisy_data_test/`](noisy_data_test/) folder, we added Gaussian noise to the grid cell activity to simulate spontaneous neural firing. Testing is also implemented in the folder. The noise is added to the grid cell activity with a different parameters, and the lifting algorithm is applied to the noisy data.

### Additional experiments

Contains scripts used to run additional experiments that are reported in the SI. Users should first generate independent trajecotires (in the 1-hole world) using `generate_1hole_trajectories.py`

## Authors

- Yuxing Jared Yao
- Iris H.R. Yoon, Ph.D.

## References

- Gardner, R. J., Hermansen, E., Pachitariu, M., Burak, Y., Baas, N. A., Dunn, B. A., ... & Moser, E. I. (2022). Toroidal topology of population activity in grid cells. *Nature, 602*(7895), 123-128.
- Wen, J. H., Sorscher, B., Aery Jones, E. A., Ganguli, S., & Giocomo, L. M. (2024). One-shot entorhinal maps enable flexible navigation in novel environments. *Nature, 635*(8040), 943-950.
- Campbell, M. G., Attinger, A., Ocko, S. A., Ganguli, S., & Giocomo, L. M. (2021). Distance-tuned neurons drive specialized path integration calculations in medial entorhinal cortex. Cell reports, 36(10).
