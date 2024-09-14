import os
import sys
import torch

main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(main_path)
print(main_path)

from MambaCD.changedetection.script import train_MambaSCD

configs_path = os.path.join(main_dir, 'MambaCD/changedetection/configs/vssm1/vssm_small_224.yaml')
model_path = os.path.join(main_dir, 'MambaCD/saved_models')

SECOND_DATASET_PATH = os.path.join(main_dir, 'ChangeDetection', 'Datasets', 'SECOND')
SECOND_TRAIN_DATASET_PATH = os.path.join(SECOND_DATASET_PATH, 'train')
SECOND_TEST_DATASET_PATH = os.path.join(SECOND_DATASET_PATH, 'test')
SECOND_TRAIN_DATA_LIST_PATH = os.path.join(SECOND_DATASET_PATH, 'train.txt')
SECOND_TEST_DATA_LIST_PATH = os.path.join(SECOND_DATASET_PATH, 'test.txt')

train_data_list = []
with open(SECOND_TRAIN_DATA_LIST_PATH, 'r') as f:
    for line in f:
        train_data_list.append(line.strip())

test_data_list = []
with open(SECOND_TEST_DATA_LIST_PATH, 'r') as f:
    for line in f:
        test_data_list.append(line.strip())

class ARGS:
    def __init__(self):
        self.cfg = configs_path
        self.opts = None
        self.pretrained_weight_path = None
        self.dataset = 'SECOND'
        self.type = 'train'
        self.train_dataset_path = SECOND_TRAIN_DATASET_PATH
        self.train_data_list_path = SECOND_TRAIN_DATA_LIST_PATH
        self.test_dataset_path = SECOND_TEST_DATASET_PATH
        self.test_data_list_path = SECOND_TEST_DATA_LIST_PATH
        self.shuffle = True
        self.batch_size = 1
        self.crop_size = 256
        self.train_data_name_list = train_data_list
        self.test_data_name_list = test_data_list
        self.start_iter = 0
        self.cuda = True
        self.max_iters = 800000
        self.model_type = 'MambaSCD'
        self.model_param_path = model_path
        self.resume = None
        self.learning_rate = 1e-4
        self.momentum = 0.9
        self.weight_decay = 5e-4

args = ARGS()

torch.cuda.empty_cache()
trainer_SECOND = train_MambaSCD.Trainer(args)
trainer_SECOND.training()
trainer_SECOND.validation()