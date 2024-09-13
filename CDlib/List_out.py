import sys
import os
import shutil
from tqdm import tqdm

main_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(main_path)

DATASET_PATH = os.path.join(main_path, 'Datasets')

def make_dir(dir_name,folder: str):
    folder_dir = os.path.join(DATASET_PATH, folder)
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)
    train_dir = os.path.join(folder_dir, dir_name)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    T1_dir = os.path.join(train_dir, 'T1')
    T2_dir = os.path.join(train_dir, 'T2')
    GT_dir = os.path.join(train_dir, 'GT')
    if not os.path.exists(T1_dir):
        os.makedirs(T1_dir)
    if not os.path.exists(T2_dir):
        os.makedirs(T2_dir)
    if not os.path.exists(GT_dir):
        os.makedirs(GT_dir)
    return T1_dir, T2_dir, GT_dir, folder_dir

def make_list_train_test(T1_dir, name, folder_dir):
    files = []
    for file in os.listdir(T1_dir):
        files.append(file)
    with open (os.path.join(folder_dir, name), 'w+') as f:
        for item in files:
            f.write("%s\n" % item)

SECOND_TRAIN_PATH_T1 = os.path.join(DATASET_PATH,'SECOND' ,'train/T1')
SECOND_TRAIN_PATH_T2 = os.path.join(DATASET_PATH,'SECOND' ,'train/T2')
SECOND_LABEL_PATH = os.path.join(DATASET_PATH,'SECOND' ,'train/GT_CD')

SECOND_TEST_PATH_T1 = os.path.join(DATASET_PATH,'SECOND' ,'test/T1')
SECOND_TEST_PATH_T2 = os.path.join(DATASET_PATH,'SECOND' ,'test/T2')
SECOND_TEST_LABEL_PATH = os.path.join(DATASET_PATH,'SECOND' ,'test/GT_CD')

class SECOND:
    def __init__(self):
        print('Processing SECOND dataset...')

    def update_bar(self, total_files, desc, original_path, new_path):
        with tqdm(total=total_files, desc= desc, unit= 'file') as pbar:
            for file in os.listdir(original_path):
                shutil.copy(os.path.join(original_path, file), os.path.join(new_path, file))
                pbar.update(1)

    def list_out_SECOND(self):
        T1_dir, T2_dir, GT_dir, folder_dir = make_dir('train','SECOND-processed')
        total_files_T1 = len(os.listdir(SECOND_TRAIN_PATH_T1))
        self.update_bar(total_files_T1, "Processing T1 in train Dataset", SECOND_TRAIN_PATH_T1, T1_dir)
        self.update_bar(total_files_T1, "Processing T2 in train Dataset", SECOND_TRAIN_PATH_T2, T2_dir)
        self.update_bar(total_files_T1, "Processing GT in train Dataset", SECOND_LABEL_PATH, GT_dir)

        make_list_train_test(T1_dir, 'train.txt', folder_dir)

        T1_dir, T2_dir, GT_dir, folder_dir = make_dir('test','SECOND-processed')
        self.update_bar(total_files_T1, "Processing T1 in test Dataset", SECOND_TEST_PATH_T1, T1_dir)
        self.update_bar(total_files_T1, "Processing T2 in test Dataset", SECOND_TEST_PATH_T2, T2_dir)
        self.update_bar(total_files_T1, "Processing GT in test Dataset", SECOND_TEST_LABEL_PATH, GT_dir)

        make_list_train_test(T1_dir, 'test.txt', folder_dir)


LEVIR_TRAIN_PATH_T1 = os.path.join(DATASET_PATH,'LEVIR-CD' ,'train/A')
LEVIR_TRAIN_PATH_T2 = os.path.join(DATASET_PATH,'LEVIR-CD' ,'train/B')
LEVIR_TRAIN_LABEL_PATH = os.path.join(DATASET_PATH,'LEVIR-CD' ,'train/label')

LEVIR_TEST_PATH_T1 = os.path.join(DATASET_PATH,'LEVIR-CD' ,'test/A')
LEVIR_TEST_PATH_T2 = os.path.join(DATASET_PATH,'LEVIR-CD' ,'test/B')
LEVIR_TEST_LABEL_PATH = os.path.join(DATASET_PATH,'LEVIR-CD' ,'test/label')

class LEVIR_CD:
    def __init__(self):
        print('Processing LEVIR-CD dataset...')

    def update_bar(self, total_files, desc, original_path, new_path):
        with tqdm(total=total_files, desc= desc, unit= 'file') as pbar:
            for file in os.listdir(original_path):
                name = file.split('_')[1]
                num = name.split('.')[0]
                name = num.zfill(5) + '.png'
                shutil.copy(os.path.join(original_path, file), os.path.join(new_path, name))
                pbar.update(1)

    def list_out_LEVIR(self):
        T1_dir, T2_dir, GT_dir, folder_dir = make_dir('train','LEVIR-processed')
        total_files_T1 = len(os.listdir(LEVIR_TRAIN_PATH_T1))
        self.update_bar(total_files_T1, "Processing T1 in train Dataset", LEVIR_TRAIN_PATH_T1, T1_dir)
        self.update_bar(total_files_T1, "Processing T2 in train Dataset", LEVIR_TRAIN_PATH_T2, T2_dir)
        self.update_bar(total_files_T1, "Processing GT in train Dataset", LEVIR_TRAIN_LABEL_PATH, GT_dir)

        make_list_train_test(T1_dir, 'train.txt', folder_dir)

        T1_dir, T2_dir, GT_dir, folder_dir = make_dir('test','LEVIR-processed')
        self.update_bar(total_files_T1, "Processing T1 in test Dataset", LEVIR_TEST_PATH_T1, T1_dir)
        self.update_bar(total_files_T1, "Processing T2 in test Dataset", LEVIR_TEST_PATH_T2, T2_dir)
        self.update_bar(total_files_T1, "Processing GT in test Dataset", LEVIR_TEST_LABEL_PATH, GT_dir)

        make_list_train_test(T1_dir, 'test.txt', folder_dir)


SYSU_TRAIN_PATH_T1 = os.path.join(DATASET_PATH,'SYSU' ,'train/train/time1')
SYSU_TRAIN_PATH_T2 = os.path.join(DATASET_PATH,'SYSU' ,'train/train/time2')
SYSU_TRAIN_LABEL_PATH = os.path.join(DATASET_PATH,'SYSU' ,'train/train/label')

SYSU_TEST_PATH_T1 = os.path.join(DATASET_PATH,'SYSU' ,'test/test/time1')
SYSU_TEST_PATH_T2 = os.path.join(DATASET_PATH,'SYSU' ,'test/test/time2')
SYSU_TEST_LABEL_PATH = os.path.join(DATASET_PATH,'SYSU' ,'test/test/label')

class SYSU_CD:
    def __init__(self):
        print('Processing SYSU-CD dataset...')

    def update_bar(self, total_files, desc, original_path, new_path):
        with tqdm(total=total_files, desc= desc, unit= 'file') as pbar:
            for file in os.listdir(original_path):
                shutil.copy(os.path.join(original_path, file), os.path.join(new_path, file))
                pbar.update(1)

    def list_out_SYSU(self):
        T1_dir, T2_dir, GT_dir, folder_dir = make_dir('train','SYSU-processed')
        total_files_T1 = len(os.listdir(SYSU_TRAIN_PATH_T1))
        self.update_bar(total_files_T1, "Processing T1 in train Dataset", SYSU_TRAIN_PATH_T1, T1_dir)
        self.update_bar(total_files_T1, "Processing T2 in train Dataset", SYSU_TRAIN_PATH_T2, T2_dir)
        self.update_bar(total_files_T1, "Processing GT in train Dataset", SYSU_TRAIN_LABEL_PATH, GT_dir)

        make_list_train_test(T1_dir, 'train.txt', folder_dir)

        T1_dir, T2_dir, GT_dir, folder_dir = make_dir('test','SYSU-processed')
        self.update_bar(total_files_T1, "Processing T1 in test Dataset", SYSU_TEST_PATH_T1, T1_dir)
        self.update_bar(total_files_T1, "Processing T2 in test Dataset", SYSU_TEST_PATH_T2, T2_dir)
        self.update_bar(total_files_T1, "Processing GT in test Dataset", SYSU_TEST_LABEL_PATH, GT_dir)

        make_list_train_test(T1_dir, 'test.txt', folder_dir)

if __name__ == '__main__':
    print("Select Dataset")
    print("1. SECOND")
    print("2. LEVIR-CD")
    print("3. SYSU-CD")
    choice = int(input())
    if choice == 1:
        SECOND().list_out_SECOND()
    elif choice == 2:
        LEVIR_CD().list_out_LEVIR()
    elif choice == 3:
        SYSU_CD().list_out_SYSU()
    else:
        print("Invalid Choice")
