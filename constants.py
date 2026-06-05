import os, sys
from pathlib import Path
from seaborn import color_palette # type: ignore
from ROOT_PATH import DATA_ROOT, REPO_PATH

sys.path.append('.')

GRID_FIELDS_PATH = os.path.join(DATA_ROOT, 'grid_fields')
PLACE_FIELDS_PATH = os.path.join(DATA_ROOT, 'place_fields')
GRID_FIG_PATH = os.path.join(REPO_PATH, 'grid_fields')
PLACE_FIG_PATH = os.path.join(REPO_PATH, 'place_fields')

GARDNER_DATA_PATH = os.path.join(DATA_ROOT, 'Data', 'Gardner_data/')
GARDNER_CONJUNCTIVE_PATH = os.path.join(GARDNER_DATA_PATH, 'is_conjunctive_all.npz')

NOISY_DATA_PATH = os.path.join(DATA_ROOT, 'noisy_data_test', 'noisy')
NOISY_DATA_FIG_PATH = os.path.join(REPO_PATH, 'noisy_data_test', 'noisy')

SUBTRACT_GAUSSIAN_DATA_PATH = os.path.join(DATA_ROOT, 'noisy_data_test', 'subtract_gaussian')
SUBTRACT_GAUSSIAN_FIG_PATH = os.path.join(REPO_PATH, 'noisy_data_test', 'subtract_gaussian')

WEN_DATA_PATH = os.path.join(DATA_ROOT, 'Data', 'Wen_et_al_data')
WEN_DATA_FIG_PATH = os.path.join(REPO_PATH, 'Wen_data_test', 'figures')

GIOMOCO_DATA_PATH = os.path.join(DATA_ROOT, 'Data', 'giomoco_data')

AFFINE_TRANSFORM_PATH = os.path.join(DATA_ROOT, 'affine_transform')
AFFINE_TRANSFORM_FIG_PATH = os.path.join(REPO_PATH, 'affine_transform')

TRAJ_PATH = os.path.join(REPO_PATH, 'trajectory')

GARDNER_TEST_FIG_PATH = os.path.join(REPO_PATH, 'Gardner_data_test', 'figures')

MANU_FIG_MATERIAL_PATH = os.path.join(REPO_PATH, '_generate_figs', 'figs')
MANU_FIG_CACHE_PATH = os.path.join(DATA_ROOT, '_generate_figs', '.cache')
DEFAULT_COLOR = color_palette('muted')[0]

# Allows for easy checking of all the paths
if __name__ == '__main__':
    missing_dirs = []
    for path_key in list(globals().keys()):
        try:    
            path = globals()[path_key]
            if os.path.exists(path) or not path_key.isupper(): continue
        except: continue
        if '.' in os.path.basename(path)[1:-1]:
            print(f'Missing file: {path}')
        else:
            print(f'Missing directory: {path}')
            missing_dirs.append(path)
    if missing_dirs and input('Create missing directories? (y/n): ') == 'y':
        for path in missing_dirs:
            os.makedirs(path)