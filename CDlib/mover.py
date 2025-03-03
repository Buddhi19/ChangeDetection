import os
import sys
import random

main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(main_dir)

TEST_FOLDER = os.path.join(main_dir,"Datasets","SECOND","test")
TRAIN_FOLDER = os.path.join(main_dir,"Datasets","SECOND","train")

def MOVER():
    """
    This is an one-time function to move 1000 images from test to train folder
    """
    all_files = os.listdir(
        os.path.join(TEST_FOLDER,"GT_T1")
    )

    selected_files = random.sample(all_files, 700)

    for file in selected_files:
        os.rename(
            os.path.join(TEST_FOLDER,"GT_T1",file),
            os.path.join(TRAIN_FOLDER,"GT_T1",file)
        )
        os.rename(
            os.path.join(TEST_FOLDER,"GT_T2",file),
            os.path.join(TRAIN_FOLDER,"GT_T2",file)
        )
        os.rename(
            os.path.join(TEST_FOLDER,"T1",file),
            os.path.join(TRAIN_FOLDER,"T1",file)
        )
        os.rename(
            os.path.join(TEST_FOLDER,"T2",file),
            os.path.join(TRAIN_FOLDER,"T2",file)
        )
        os.rename(
            os.path.join(TEST_FOLDER,"GT_CD",file),
            os.path.join(TRAIN_FOLDER,"GT_CD",file)
        )
        os.rename(
            os.path.join(TEST_FOLDER,"GT_T1_COLORED",file),
            os.path.join(TRAIN_FOLDER,"GT_T1_COLORED",file)
        )
        os.rename(
            os.path.join(TEST_FOLDER,"GT_T2_COLORED",file),
            os.path.join(TRAIN_FOLDER,"GT_T2_COLORED",file)
        )
    return

def list_out():
    f_all_train = os.listdir(os.path.join(TRAIN_FOLDER,"GT_T1"))
    train_file = os.path.join(main_dir,"Datasets","SECOND","train.txt")
    with open(train_file,"w") as f:
        for file in f_all_train:
            f.write(file+"\n")
    
    f_all_test = os.listdir(os.path.join(TEST_FOLDER,"GT_T1"))
    test_file = os.path.join(main_dir,"Datasets","SECOND","test.txt")
    with open(test_file,"w") as f:
        for file in f_all_test:
            f.write(file+"\n")
    return
    

if __name__ == "__main__":
    MOVER()
    list_out()
