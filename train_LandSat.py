import os
import sys
import torch
from dotenv import load_dotenv

load_dotenv()

def getPath(env_path):
    return os.path.expanduser(os.getenv(env_path))

VSSM_MODEL_PATH = getPath('VSSMBASEPATH')

LandSat_DATASET_PATH = getPath('LANDSAT')
print(LandSat_DATASET_PATH)
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

print(f"Number of training samples: {len(train_data_list)}")

torch.cuda.set_device(1)
main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(main_dir)
print(main_dir)

from RemoteSensing.changedetection.script import train_MambaSCD 

configs_path = os.path.join(main_dir, 'RemoteSensing/changedetection/configs/vssm1/vssm_base_224.yaml')
model_path = os.path.join(main_dir, 'RemoteSensing/saved_models')

class ARGS:
    def __init__(self):
        self.cfg = configs_path
        self.opts = None
        self.pretrained_weight_path = None
        self.dataset = 'LandSat'
        self.type = 'train'
        self.train_dataset_path = LandSat_DATASET_PATH
        self.train_data_list_path = LandSat_TRAIN_DATA_LIST_PATH
        self.test_dataset_path = LandSat_DATASET_PATH
        self.test_data_list_path = LandSat_TEST_DATA_LIST_PATH
        self.shuffle = True
        self.batch_size = 4
        self.crop_size = 416
        self.train_data_name_list = train_data_list
        self.test_data_name_list = test_data_list
        self.start_iter = 0
        self.cuda = True
        self.max_iters = 1600000
        self.model_type = 'MambaSCD_base'
        self.model_param_path = model_path
        self.resume = None
        self.learning_rate = 1e-4
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.num_classes = 5

args = ARGS()

torch.cuda.empty_cache()
trainer_LandSat = train_MambaSCD.Trainer(args)
trainer_LandSat.training()
trainer_LandSat.validation()