# import cv2
# image = cv2.imread('E:\\train_data\\trash_dataset\\data\\test_images\\cardboard262.jpg',1)
# resized = cv2.resize(image, (224, 224))
# cv2.imwrite('F:\\resized_image.jpeg', resized)
# cv2.imshow('image',image)
# cv2.waitKey(0)
# # cv2.destroyAllWindows()
# import os
# path='E:\\train_data\\trash_dataset\\all_data\\'
# image_list=os.listdir(path)
# print(len(image_list))
# for i in range(len(image_list)):
#     print(image_list[i])
# class_correct = list(0 for i in range(5))
# print(class_correct)
# import cv2 as cv
# import numpy as np
#
# #椒盐噪声在图像上添加随机的黑白点
# def add_salt_pepper_noise(image):
#     h, w = image.shape[:2]
#     nums = 5000
#     rows = np.random.randint(0, h, nums, dtype=np.int)
#     cols = np.random.randint(0, w, nums, dtype=np.int)
#     for i in range(nums):
#         if i % 2 == 1:
#             image[rows[i], cols[i]] = (255, 255, 255)
#         else:
#             image[rows[i], cols[i]] = (0, 0, 0)
#     return image
#
# #高斯噪声
# def gaussian_noise(image):
#     noise = np.zeros(image.shape, image.dtype)
#     m = (15, 15, 15)
#     s = (30, 30, 30)
#     cv.randn(noise, m, s)
#     dst = cv.add(image, noise)
#     # cv.imshow("gaussian noise", dst)
#     # cv.waitKey(0)
#     return dst
#
# image = cv.imread('E:\\train_data\\trash_dataset\\data\\train_images\\organism3607.jpg')
# cv.imshow('originalImage',image)
# cv.imwrite('F:/originalImage.jpg',image)
# cv.waitKey(0)
# image_one= gaussian_noise(image)
# cv.imwrite('F:/salt_pepper_Image.jpg',image_one)
# image_two=add_salt_pepper_noise(image)
# cv.imwrite('F:/salt_pepper.jpg',image_two)
# cv.imshow('salt_pepper_Image',image_two)
# cv.waitKey(0)
# cv.destroyAllWindows()
# from torchvision import transforms, datasets
# #
# transformer = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.299, 0.224, 0.255])
# ])
# test_dataset = datasets.ImageFolder(root='E:/train_data/trash_dataset/new_data/test_data/', transform=transformer)
# print(test_dataset.class_to_idx)
# import sys
# print(sys.getdefaultencoding())  # 默认的编码方式为UTF-8
# s = '中文显示的测试'
# print(s)
import chardet # 编码检测模块
# with open('F:/demo.txt', 'r',encoding='UTF-8') as f:
#     '''
#     UnicodeDecodeError: 'gbk' codec can't decode byte 0x80
#     in position 8: illegal multibyte sequence
#     '''
#
#     content = f.read()
#     # type = chardet.detect(content)
#     # content1 = content.decode(type["encoding"])
#     print(content)
import numpy as np
arr = np.random.permutation(10)
print(arr) #[7 1 4 5 2 6 9 0 3 8]