import os
import sys

main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(main_dir)

from dotenv import load_dotenv

load_dotenv()
def getPath(env_path):
    return os.path.expanduser(os.getenv(env_path))

LANDSAT_DATASET_PATH = os.path.abspath('/storage/scratch3/buddhiw-change-detection/Datasets/Landsat-SCD_dataset/')

TIME_1 = os.path.join(LANDSAT_DATASET_PATH, 'A')
TIME_2 = os.path.join(LANDSAT_DATASET_PATH, 'B')

Labels = os.path.join(LANDSAT_DATASET_PATH, 'label')

total_files = []
for files in os.listdir(TIME_1):
    if files.endswith('.png'):
        total_files.append(files)

train_files = total_files[:int(len(total_files) * 0.8)]
test_files = total_files[int(len(total_files) * 0.8):]

def write_to_files(file_name, file_list):
    with open(file_name, 'w') as f:
        for file in file_list:
            f.write(f"{file}\n")

if __name__ == "__main__":
    path = os.path.join(LANDSAT_DATASET_PATH, 'train.txt')
    write_to_files(path, train_files)
    path = os.path.join(LANDSAT_DATASET_PATH, 'test.txt')
    write_to_files(path, test_files)