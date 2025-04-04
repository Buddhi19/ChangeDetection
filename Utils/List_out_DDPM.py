import os
import sys

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(MAIN_DIR)

SECOND_TRAIN_PATH_T1 = os.path.join(MAIN_DIR, 'Datasets', 'SECOND', 'train/T1')
SECOND_TRAIN_PATH_T2 = os.path.join(MAIN_DIR, 'Datasets', 'SECOND', 'train/T2')

SYSU_TRAIN_PATH_T1 = os.path.join(MAIN_DIR, 'Datasets', 'SYSU', 'train/time1')
SYSU_TRAIN_PATH_T2 = os.path.join(MAIN_DIR, 'Datasets', 'SYSU', 'train/time2')

with open(os.path.join(MAIN_DIR, 'Datasets', 'DDPM_TOTAL', 'train.txt'), 'w+') as f:
    for file in os.listdir(SECOND_TRAIN_PATH_T1):
        f.write(f"Datasets/SECOND/train/T1/{file}\n")
    for file in os.listdir(SYSU_TRAIN_PATH_T1):
        f.write(f"Datasets/SYSU/train/time1/{file}\n")
    for file in os.listdir(SECOND_TRAIN_PATH_T2):
        f.write(f"Datasets/SECOND/train/T2/{file}\n")
    for file in os.listdir(SYSU_TRAIN_PATH_T2):
        f.write(f"Datasets/SYSU/train/time2/{file}\n")