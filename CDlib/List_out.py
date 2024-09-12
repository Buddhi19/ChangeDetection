import sys
import os

main_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(main_path)

SECOND_DATASET_PATH = os.path.join(main_path, 'Datasets')

TRAIN_PATH_T1 = os.path.join(SECOND_DATASET_PATH,'SECOND' ,'train/T1')
TRAIN_PATH_T2 = os.path.join(SECOND_DATASET_PATH,'SECOND' ,'train/T2')
LABEL_PATH = os.path.join(SECOND_DATASET_PATH,'SECOND' ,'train/GT_CD')

LEVIR_TRAIN_PATH_T1 = os.path.join(SECOND_DATASET_PATH,'LEVIR-CD' ,'train/A')
LEVIR_TRAIN_PATH_T2 = os.path.join(SECOND_DATASET_PATH,'LEVIR-CD' ,'train/B')
LEVIR_TRAIN_LABEL_PATH = os.path.join(SECOND_DATASET_PATH,'LEVIR-CD' ,'train/label')

LEVIR_TEST_PATH_T1 = os.path.join(SECOND_DATASET_PATH,'LEVIR-CD' ,'test/A')
LEVIR_TEST_PATH_T2 = os.path.join(SECOND_DATASET_PATH,'LEVIR-CD' ,'test/B')
LEVIR_TEST_LABEL_PATH = os.path.join(SECOND_DATASET_PATH,'LEVIR-CD' ,'test/label')

def list_out_SECOND():
    files_T1 = []
    for file in os.listdir(TRAIN_PATH_T1):
        files_T1.append(file)
    files_T2 = []
    for file in os.listdir(TRAIN_PATH_T2):
        files_T2.append(file)
    files_LABEL = []
    for file in os.listdir(LABEL_PATH):
        files_LABEL.append(file)

    T1_OUT = os.path.join(SECOND_DATASET_PATH, 'T1.txt')
    T2_OUT = os.path.join(SECOND_DATASET_PATH, 'T2.txt')
    LABEL_OUT = os.path.join(SECOND_DATASET_PATH, 'GT_CD.txt')

    with open(T1_OUT, 'w+') as f:
        for item in files_T1:
            f.write("%s\n" % item)

    with open(T2_OUT, 'w+') as f:
        for item in files_T2:
            f.write("%s\n" % item)

    with open(LABEL_OUT, 'w+') as f:
        for item in files_LABEL:
            f.write("%s\n" % item)

def list_out_LEVIR():
    folder_dir = os.path.join(SECOND_DATASET_PATH, 'LEVIR-processed')
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)

    def make_dir(dir_name):
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
        return T1_dir, T2_dir, GT_dir

    T1_dir, T2_dir, GT_dir = make_dir('train')
    for file in os.listdir(LEVIR_TRAIN_PATH_T1):
        name = file.split('_')[1]
        name = name.zfill(5)
        os.rename(os.path.join(LEVIR_TRAIN_PATH_T1, file), os.path.join(T1_dir, name))
    for file in os.listdir(LEVIR_TRAIN_PATH_T2):
        name = file.split('_')[1]
        name = name.zfill(5)
        os.rename(os.path.join(LEVIR_TRAIN_PATH_T2, file), os.path.join(T2_dir, name))
    for file in os.listdir(LEVIR_TRAIN_LABEL_PATH):
        name = file.split('_')[1]
        name = name.zfill(5)
        os.rename(os.path.join(LEVIR_TRAIN_LABEL_PATH, file), os.path.join(GT_dir, name))

    files_train = []
    for file in os.listdir(T1_dir):
        files_train.append(file)

    with open (os.path.join(folder_dir, 'train.txt'), 'w+') as f:
        for item in files_train:
            f.write("%s\n" % item)

    T1_dir, T2_dir, GT_dir = make_dir('test')
    for file in os.listdir(LEVIR_TEST_PATH_T1):
        name = file.split('_')[1]
        name = name.zfill(5)
        os.rename(os.path.join(LEVIR_TEST_PATH_T1, file), os.path.join(T1_dir, name))
    for file in os.listdir(LEVIR_TEST_PATH_T2):
        name = file.split('_')[1]
        name = name.zfill(5)
        os.rename(os.path.join(LEVIR_TEST_PATH_T2, file), os.path.join(T2_dir, name))
    for file in os.listdir(LEVIR_TEST_LABEL_PATH):
        name = file.split('_')[1]
        name = name.zfill(5)
        os.rename(os.path.join(LEVIR_TEST_LABEL_PATH, file), os.path.join(GT_dir, name))

    files_test = []
    for file in os.listdir(T1_dir):
        files_test.append(file)
    
    with open (os.path.join(folder_dir, 'test.txt'), 'w+') as f:
        for item in files_test:
            f.write("%s\n" % item)
    
    


if __name__ == '__main__':
    list_out_LEVIR()
