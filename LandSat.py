import os
import sys
import torch

main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(main_dir)
print(main_dir)

from RemoteSensing.changedetection.script import train_MambaSCD_landsat

torch.cuda.empty_cache()
torch.cuda.set_device(0)

configs_path = os.path.join(main_dir, 'RemoteSensing/changedetection/configs/vssm1/vssm_base_224.yaml')
VSSM_MODEL_PATH = os.path.join(
    main_dir, 'MambaCD/pretrained/vssm_base_0229_ckpt_epoch_237.pth'
)
model_path = os.path.join(main_dir, 'RemoteSensing/saved_models')
model_path_trained = os.path.join(main_dir,'RemoteSensing/saved_models/LandSat2/31500_model.pth')

LandSat_DATASET_PATH = os.path.join(main_dir, 'Datasets', 'Landsat-SCD')
LandSat_TRAIN_DATA_LIST_PATH = os.path.join(LandSat_DATASET_PATH, 'train_list.txt')
LandSat_TEST_DATA_LIST_PATH = os.path.join(LandSat_DATASET_PATH, 'test_list.txt')

train_data_list = []
with open(LandSat_TRAIN_DATA_LIST_PATH, 'r') as f:
    for line in f:
        train_data_list.append(line.strip())

test_data_list = []
with open(LandSat_TEST_DATA_LIST_PATH, 'r') as f:
    for line in f:
        test_data_list.append(line.strip())

class ARGS:
    def __init__(self):
        self.cfg = configs_path
        self.opts = None
        self.pretrained_weight_path = VSSM_MODEL_PATH
        self.dataset = 'LandSat'
        self.type = 'train'
        self.train_dataset_path = LandSat_DATASET_PATH
        self.train_data_list_path = LandSat_TRAIN_DATA_LIST_PATH
        self.test_dataset_path = LandSat_DATASET_PATH
        self.test_data_list_path = LandSat_TEST_DATA_LIST_PATH
        self.shuffle = True
        self.batch_size = 4
        self.crop_size = 256
        self.train_data_name_list = train_data_list
        self.test_data_name_list = test_data_list
        self.start_iter = 31500
        self.cuda = True
        self.max_iters = 800000
        self.model_type = 'MambaSCD_Base'
        self.model_param_path = model_path
        self.resume = model_path_trained
        self.learning_rate = 1e-4
        self.momentum = 0.9
        self.weight_decay = 5e-4

args = ARGS()

torch.cuda.empty_cache()
trainer_LandSat = train_MambaSCD_landsat.Trainer(args)
trainer_LandSat.training()
trainer_LandSat.validation()