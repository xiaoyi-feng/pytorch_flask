'''
step_1: rename the images files in each folder
step_2: split the index of images for 4:1
step_3: resize the images
final result is like this:
data
  |--train
      |--类别1
      |--类别2
      ...
  |--test
      |--类别1
      |--类别2
      ...
'''


import os

class BatchRename():
    '''
    批量重命名文件夹中的图片文件
    '''

    def __init__(self, path,image_prefix):  # path 是图片所在文件夹的路径
        self.path = path
        self.image_prefix=image_prefix

    '''按照文件夹名称的后半部分来给该文件夹下的图片命名'''
    def rename(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)
        i = 0
        for item in filelist:
            #if item.endswith('.jpeg') or item.endswith('.jpg'):
                #print(item)
            src = os.path.join(os.path.abspath(self.path), item)
            dst = os.path.join(os.path.abspath(self.path), self.image_prefix+str(i) + '.jpeg')
            #dst = os.path.join(os.path.abspath(self.path), str(i) + '_.jpeg')

            os.rename(src, dst)
            #print('converting %s to %s ...' % (src, dst))
            i = i + 1

        print('total %d to rename & converted %d jpegs' % (total_num, i))


if __name__ == '__main__':
    # prefix_path='E:\\train_data\\trash_dataset\\all_data\\'
    # waste_type = os.listdir(prefix_path)
    # for i in range(len(waste_type)):
    #     whole_name = waste_type[i]
    #     postfix_name = whole_name[2:]
    #     print(postfix_name)
    #     demo = BatchRename(prefix_path+whole_name,postfix_name)
    #     demo.rename()
    demo = BatchRename('E:\\train_data\\trash_dataset\\all_data\\H_thermometer', 'thermometer')
    demo.rename()