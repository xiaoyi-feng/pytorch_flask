import cv2
import numpy as np
import os

#椒盐噪声:在图像上添加5000个随机的黑白点
def add_salt_pepper_noise(image):
    h, w = image.shape[:2]
    nums = 500
    rows = np.random.randint(0, h, nums, dtype=np.int)
    cols = np.random.randint(0, w, nums, dtype=np.int)
    for i in range(nums):
        if i % 2 == 1:
            image[rows[i], cols[i]] = (255, 255, 255)
        else:
            image[rows[i], cols[i]] = (0, 0, 0)
    return image

#高斯噪声
def gaussian_noise(image):
    noise = np.zeros(image.shape, image.dtype)
    m = (15, 15, 15)
    s = (30, 30, 30)
    cv2.randn(noise, m, s) # 生成一个与原始图片大小相等的矩阵，矩阵中的值在【15,30】之间
    dst = cv2.add(image, noise)
    return dst

# 对图片进行旋转
def rotate(image, angle, center=None, scale=1.0): #1
    (h, w) = image.shape[:2] #2
    if center is None: #3
        center = (w // 2, h // 2) #4

    M = cv2.getRotationMatrix2D(center, angle, scale) #5

    rotated = cv2.warpAffine(image, M, (w, h)) #6
    return rotated #7


def enlarge_the_scale():
    path='E:\\train_data\\trash_dataset\\all_data\\'
    image_list=os.listdir(path)
    print(len(image_list))
    for i in range(len(image_list)):
        each_folder_path=os.path.join(path, image_list[i] +'\\')
        print(each_folder_path) # E:\train_data\trash_dataset\all_data\D_adhesiveTape\
        each_folder_list= os.listdir(each_folder_path)
        length = len(each_folder_list)  # 每个文件夹下的image数目

        if length<300:
            print('before code length is {}'.format(length))
            for j in range(length):
                each_image_path = os.path.join(each_folder_path,each_folder_list[j])
                each_image = cv2.imread(each_image_path)

                rotate_img_90=rotate(each_image,90)
                saved_path_90 = 'E:\\train_data\\trash_dataset\\all_data\\'+image_list[i] + '\\rot_90_' +str(j)+'.jpeg'
                cv2.imwrite(saved_path_90, rotate_img_90)

                rotate_img_180 = rotate(each_image, 180)
                saved_path_180 = 'E:\\train_data\\trash_dataset\\all_data\\' + image_list[i] + '\\rot_180_' + str(
                    j) + '.jpeg'
                cv2.imwrite(saved_path_180, rotate_img_180)

                gaussi_img = gaussian_noise(each_image)
                saved_path_gaussi = 'E:\\train_data\\trash_dataset\\all_data\\' + image_list[i] + '\\gaussi_' + str(
                    j) + '.jpeg'
                cv2.imwrite(saved_path_gaussi, gaussi_img)

                salt_img = add_salt_pepper_noise(each_image)
                saved_path_salt = 'E:\\train_data\\trash_dataset\\all_data\\' + image_list[i] + '\\salt_' + str(
                    j) + '.jpeg'
                cv2.imwrite(saved_path_salt, salt_img)

            print('after code length is {}'.format(len(os.listdir(each_folder_path))))

        if length>300 and length<400:
            print('before code length is {}'.format(length))
            for j in range(length):
                each_image_path = os.path.join(each_folder_path, each_folder_list[j])
                each_image = cv2.imread(each_image_path)

                gaussi_img = gaussian_noise(each_image)
                saved_path_gaussi = 'E:\\train_data\\trash_dataset\\all_data\\' + image_list[i] + '\\gaussi_' + str(
                    j) + '.jpeg'
                cv2.imwrite(saved_path_gaussi, gaussi_img)

                salt_img = add_salt_pepper_noise(each_image)
                saved_path_salt = 'E:\\train_data\\trash_dataset\\all_data\\' + image_list[i] + '\\salt_' + str(
                    j) + '.jpeg'
                cv2.imwrite(saved_path_salt, salt_img)

            print('after code length is {}'.format(len(os.listdir(each_folder_path))))

if __name__ == '__main__':
    enlarge_the_scale()
