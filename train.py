"""
    [AUE8088] PA1: Image Classification
        - To run: (aue8088) $ python train.py
        - For better flexibility, consider using LightningCLI in PyTorch Lightning
"""
# wandb
import wandb

# PyTorch & Pytorch Lightning
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning import Trainer
import torch

# Custom packages
from src.dataset import TinyImageNetDatasetModule
from src.network import SimpleClassifier
import src.config as cfg

import os

torch.set_float32_matmul_precision('medium')


if __name__ == "__main__":
    for batch_size in cfg.BATCH_SIZE_LIST:
        num_workers = int(cfg.MEM_SIZE_MAX) / batch_size
        cfg.BATCH_SIZE = batch_size
        cfg.NUM_WORKERS = min(12, int(num_workers))
        cfg.WANDB_NAME          = f'{cfg.MODEL_NAME}-B{cfg.BATCH_SIZE}-{cfg.OPTIMIZER_PARAMS["type"]}'
        cfg.WANDB_NAME         += f'-{cfg.SCHEDULER_PARAMS["type"]}{cfg.OPTIMIZER_PARAMS["lr"]:.1E}'
        
        print("\nBatch size: ", cfg.BATCH_SIZE, "\nNum workers: ", cfg.NUM_WORKERS, "\nDB name: ", cfg.WANDB_NAME)  # Print batch_size, num_workers, and wandb_name with information

        model = SimpleClassifier(
            model_name = cfg.MODEL_NAME,
            num_classes = cfg.NUM_CLASSES,
            optimizer_params = cfg.OPTIMIZER_PARAMS,
            scheduler_params = cfg.SCHEDULER_PARAMS,
        )

        datamodule = TinyImageNetDatasetModule(
            batch_size = cfg.BATCH_SIZE,
        )

        # Read key data from a text file
        file_path = os.path.join(os.path.dirname(__file__), 'wandb.txt')
        with open(file_path, 'r') as file:
            key_data = file.read().strip()
        wandb.login(key_data=key_data)
        
        # Login to wandb with the key data
        wandb_logger= WandbLogger(
            project = cfg.WANDB_PROJECT,
            save_dir = cfg.WANDB_SAVE_DIR,
            entity = cfg.WANDB_ENTITY,
            name = cfg.WANDB_NAME,
        )

        trainer = Trainer(
            accelerator = cfg.ACCELERATOR,
            devices = cfg.DEVICES,
            precision = cfg.PRECISION_STR,
            max_epochs = cfg.NUM_EPOCHS,
            check_val_every_n_epoch = cfg.VAL_EVERY_N_EPOCH,
            logger = wandb_logger,
            callbacks = [
                LearningRateMonitor(logging_interval='epoch'),
                ModelCheckpoint(save_top_k=1, monitor='accuracy/val', mode='max'),
            ],
        )

        trainer.fit(model, datamodule=datamodule)
        trainer.validate(ckpt_path='best', datamodule=datamodule)
        
        del model
        del datamodule
        del wandb_logger
        del trainer
