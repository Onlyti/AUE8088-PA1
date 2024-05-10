import os
import numpy as np

# Training Hyperparameters
NUM_CLASSES         = 200
BATCH_SIZE_LIST     = [4096]
BATCH_SIZE          = 2048
MEM_SIZE_MAX        = 4096
VAL_EVERY_N_EPOCH   = 1

NUM_EPOCHS          = 60
OPTIMIZER_PARAMS    = {'type': 'SGD', 'lr': 0.005, 'momentum': 0.9}
milstone_rate       = np.asarray([0.75, 0.9])
SCHEDULER_PARAMS    = {'type': 'MultiStepLR', 'milestones': (NUM_EPOCHS*milstone_rate).astype(int), 'gamma': 0.5}

# Dataaset
DATASET_ROOT_PATH   = 'datasets/'
NUM_WORKERS         = 2

# Augmentation
IMAGE_ROTATION      = 20
IMAGE_FLIP_PROB     = 0.5
IMAGE_NUM_CROPS     = 64
IMAGE_PAD_CROPS     = 4
IMAGE_MEAN          = [0.4802, 0.4481, 0.3975]
IMAGE_STD           = [0.2302, 0.2265, 0.2262]

# Network
MODEL_NAME          = 'resnet18'

# Compute related
ACCELERATOR         = 'gpu'
DEVICES             = [0]
PRECISION_STR       = '32-true'

# Logging
WANDB_PROJECT       = 'aue8088-pa1'
WANDB_ENTITY        = os.environ.get('WANDB_ENTITY')
WANDB_SAVE_DIR      = 'wandb/'
WANDB_IMG_LOG_FREQ  = 50
WANDB_NAME          = f'{MODEL_NAME}-B{BATCH_SIZE}-{OPTIMIZER_PARAMS["type"]}'
WANDB_NAME         += f'-{SCHEDULER_PARAMS["type"]}{OPTIMIZER_PARAMS["lr"]:.1E}'
