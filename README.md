# GridCellDecoding

This project aims to decode trajectories from spike train data of grid cells using topological data analysis (TDA) methods. It includes complete pipeline for decoding the data and implemented validations on both simulated and experimental data.

## Table of Contents

- [GridCellDecoding](#gridcelldecoding)
  - [Table of Contents](#table-of-contents)
  - [Setup](#setup)
    - [Environment Installation](#environment-installation)
  - [Components](#components)
    - [Trajectory simulation](#trajectory-simulation)
    - [Grid cell simulation](#grid-cell-simulation)
    - [Place cell simulation (deprecated)](#place-cell-simulation-deprecated)
    - [Persistent homology](#persistent-homology)
    - [Toroidal coordinates](#toroidal-coordinates)
    - [Lifting algorithm](#lifting-algorithm)
    - [Epsilon computation](#epsilon-computation)
    - [Affine transformation \& Mismatch score](#affine-transformation--mismatch-score)
    - [Noise-adding simulation](#noise-adding-simulation)
    - [Lifting error cases](#lifting-error-cases)
    - [1D Data Test](#1d-data-test)
    - [Gardner Data Test](#gardner-data-test)
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

### Trajectory simulation

Random walk in an enclosed space is simulated in `trajectory.py`. The world can be configured with size (100*100 by default) and number of holes, inputted with `(LL_x, LL_y, UR_x, UR_y)` coordinates of the holes as a list. The simulated trajactory will be saved as `(trajectory, world)`.

### Grid cell simulation

Grid cell simulation is adopted from *Gardner et al., 2022*. Grid cells are simulated in `grid_cell_simulation.py` using a continuous attractor network (CAN) model. `utils.py` contains helper functions from the original studies.

*`Grid.py` is deprecated for an alternative simulation of grid cells.*

### Place cell simulation (deprecated)

In `place_cell_simulation.py`, the place cell simulation is implemented by generating firing rates based on randomized place fields. The activities are fitted with bivariate distribution based on the simulated trajectory, executed with `build_place_spike_train`.

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

### Affine transformation & Mismatch score

In [`affine_transformation/`](affine_transformation/) folder, `get_transform_mat.py` computes the affine transformation matrix between two paths using `estimateAffine2D` function from the OpenCV package. The affine transformation is used to align the lifted path with the original trajectory.

`score_mismatch.py` computes the mismatch score between two paths and is used to evaluate the performance of the lifting algorithm. The mismatch score is defined as the average Euclidean distance between the two paths, normalized by the size of the world. If not specified, the world size is computed as the maximum horizontal/vertical distance of path2.

### Noise-adding simulation

To test the robustness of this pipeline, we added noise to the simulated grid cell activity. In `add_noise.py`, we added Gaussian noise to the grid cell activity to simulate spontaneous neural firing. Testing is implemented in [`noisy_data_test/`](noisy_data_test/) folder. The noise is added to the grid cell activity with a different parameters, and the lifting algorithm is applied to the noisy data.

### Lifting error cases

[`lifting_error_cases/`](lifting_error_cases/) folder contains two .ipynb files that illustrate possible failures of the lifting algorithm.

### 1D Data Test

The lifting algorithm is tested on 1D data in [`data_1d_test/`](data_1d_test/) folder using grid cell data from Wen et al. (2024). The data is collected in a wheel running experiment. `Wen_1d_data_test.ipynb` applies the pipeline to the 1D data and successfully recovers the 1D trajectory.

### Gardner Data Test

The lifting algorithm is tested on 2D trajectory grid cell data in [`Gardner_data_test/`](gardner_data_test/) folder using grid cell data from Gardner et al. (2022). The data is collected in a 2D open field experiment. The lifted path is compared with the original trajectory to test the performance of the lifting algorithm.

## Authors

- Yuxing Jared Yao
- Iris H.R. Yoon, Ph.D.

## References

- Gardner, R. J., Hermansen, E., Pachitariu, M., Burak, Y., Baas, N. A., Dunn, B. A., ... & Moser, E. I. (2022). Toroidal topology of population activity in grid cells. *Nature, 602*(7895), 123-128.
- Wen, J. H., Sorscher, B., Aery Jones, E. A., Ganguli, S., & Giocomo, L. M. (2024). One-shot entorhinal maps enable flexible navigation in novel environments. *Nature, 635*(8040), 943-950.
