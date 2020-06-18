import os
import shutil
import cv2
import numpy as np

'''将所有的数据按照3:1的比例划分为train，test'''
def split_indices(datapath, test_ratio) :
    np.random.seed(42)
    # permutation 随机生成0-len(datapath)随机序列
    total_num = len(os.listdir(datapath))  # 该文件夹下图片的数目
    shuffle_indices = np.random.permutation(total_num)
    test_set_size = int(total_num * test_ratio)
    test_indices = shuffle_indices[:test_set_size]
    train_indices = shuffle_indices[test_set_size:]
    return train_indices, test_indices

## return：返回的是一个图片名称（由waste_type和对应的index组成的）的列表
def get_names(waste_type, indices):
    file_names = [waste_type + str(i) + ".jpeg" for i in indices]
    return file_names

def make_dir():
    prefix_path_train = 'E:\\train_data\\trash_dataset\\new_data\\train_data\\'
    prefix_path_test = 'E:\\train_data\\trash_dataset\\new_data\\test_data\\'
    waste_type = os.listdir('E:\\train_data\\trash_dataset\\all_data\\')

    for path in waste_type:
        train_dest_path=os.path.join(prefix_path_train, path)
        print(train_dest_path)
        if not os.path.exists(train_dest_path):
            os.makedirs(train_dest_path)

        test_dest_path= os.path.join(prefix_path_test, path)
        print(test_dest_path)
        if not os.path.exists(test_dest_path):
            os.makedirs(test_dest_path)

if __name__ =='__main__':
    make_dir()

    prefix_path_train = 'E:\\train_data\\trash_dataset\\new_data\\train_data\\'
    prefix_path_test = 'E:\\train_data\\trash_dataset\\new_data\\test_data\\'
    all_data_path= 'E:\\train_data\\trash_dataset\\all_data\\'
    waste_type = os.listdir(all_data_path)  # 每一个文件夹

    for waste in waste_type:
        source_path = os.path.join(all_data_path, waste)
        prefix_train_dst=os.path.join(prefix_path_train , waste)
        train_indices,test_indices = split_indices(source_path, 0.25)  # for each folder
        postfix_waste_type = waste[2:]
        train_image_name = get_names(postfix_waste_type, train_indices)
        print(waste)
        print(len(train_image_name))

        for i in range(len(train_image_name)): # 每一张将要移动的image，首先进行resized

            train_src = os.path.join(source_path, train_image_name[i])
            train_dst = os.path.join(prefix_train_dst, train_image_name[i])
            try:
                image = cv2.imread(train_src)
                resized = cv2.resize(image, (224, 224))
                cv2.imwrite(train_dst, resized)
            except:
                print(train_dst)

        test_image_name = get_names(postfix_waste_type, test_indices)
        prefix_test_dst = os.path.join(prefix_path_test , waste)

        for i in range(len(test_image_name)):  # 每一张将要移动的image，首先进行resized
            test_src = os.path.join(source_path, test_image_name[i])
            test_dst = os.path.join(prefix_test_dst, test_image_name[i])
            try:
                image = cv2.imread(test_src)
                resized = cv2.resize(image, (224, 224))
                cv2.imwrite(test_dst, resized)
            except:
                print(test_dst)






