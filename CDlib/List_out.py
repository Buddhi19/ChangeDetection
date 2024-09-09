import sys
import os

main_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(main_path)

SECOND_DATASET_PATH = os.path.join(main_path, 'Datasets/SECOND/SECOND')

TRAIN_PATH_T1 = os.path.join(SECOND_DATASET_PATH, 'train/T1')
TRAIN_PATH_T2 = os.path.join(SECOND_DATASET_PATH, 'train/T2')
LABEL_PATH = os.path.join(SECOND_DATASET_PATH, 'train/GT_CD')

def list_out():
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

if __name__ == '__main__':
    list_out()
