import os
import sys
import torch
from dotenv import load_dotenv

load_dotenv()
def getPath(env_path):
    return os.path.expanduser(os.getenv(env_path))

VSSM_MODEL_PATH = getPath('VSSMBASEPATH')
SECOND_DATASET_PATH = os.path.abspath('/storage/scratch3/buddhiw-change-detection/Datasets/SECOND/')


SECOND_TRAIN_DATASET_PATH = os.path.join(SECOND_DATASET_PATH, 'train')
SECOND_TEST_DATASET_PATH = os.path.join(SECOND_DATASET_PATH, 'test')
SECOND_TRAIN_DATA_LIST_PATH = os.path.join(SECOND_DATASET_PATH, 'train.txt')
SECOND_TEST_DATA_LIST_PATH = os.path.join(SECOND_DATASET_PATH, 'test.txt')


main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(main_dir)
print(main_dir)

from RemoteSensing.changedetection.script import train_MambaSCD

torch.cuda.set_device(1)

configs_path = os.path.join(main_dir, 'RemoteSensing/changedetection/configs/vssm1/vssm_base_224.yaml')

model_path = os.path.abspath('/storage/scratch3/buddhiw-change-detection/Mamba/')

STORAGE_PATH = os.path.abspath('/storage/scratch3/buddhiw-change-detection/Mamba/CA_spatial_fft_16_512/')

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
        self.pretrained_weight_path = VSSM_MODEL_PATH
        self.dataset = 'SECOND'
        self.type = 'train'
        self.train_dataset_path = SECOND_TRAIN_DATASET_PATH

        self.test_dataset_path = SECOND_TEST_DATASET_PATH

        
        self.shuffle = True
        self.batch_size = 4
        self.crop_size = 512
        self.train_data_name_list = train_data_list
        self.test_data_name_list = test_data_list
        self.start_iter = 0
        self.cuda = True
        self.max_iters = 800000
        self.model_type = 'MambaSCD_base'
        self.model_param_path = model_path

        self.resume = None 
        self.optim_path = None
        self.scheduler_path = None

        self.learning_rate = 1e-4
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.num_classes = 7
        self.model_saving_name = 'SECOND'
        

args = ARGS()

torch.cuda.empty_cache()

trainer_SECOND = train_MambaSCD.Trainer(args)
trainer_SECOND.training()