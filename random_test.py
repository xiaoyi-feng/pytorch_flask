import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision import transforms, datasets
import cv2
import os
import random

if __name__ =='__main__':
    # BATCH_SIZE = 8
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')     # 使用GPU


    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.299, 0.224, 0.255])
    ])

    #classes = ('cardboard','glass','metal','organism','paper','plastic')
    #classes = os.listdir('E:/train_data/trash_dataset/new_data/test_data/')
    classes = ('D_adhesiveTape', 'D_cottonswabs', 'D_faceMask', 'D_featherDuster', 'D_flyswatter',
               'D_foodBox', 'D_glassesCloth', 'D_makerPen', 'D_napkin', 'D_strawHat', 'D_toothBrush', 'D_towel',
               'H_battery', 'H_drycell', 'H_lamp', 'H_pesticide', 'H_pills', 'H_solarPanel', 'H_thermometer',
               'R_TV', 'R_TrolleyCase', 'R_bag', 'R_bedding', 'R_calculator', 'R_cardboard', 'R_cellPhone',
               'R_chargeLine', 'R_desk', 'R_desklamp', 'R_earphone', 'R_fanner', 'R_fireExtinguisher',
               'R_glass', 'R_keyboard', 'R_magicMouse', 'R_magneticCard',
               'R_metal', 'R_microphone', 'R_paper', 'R_plastic', 'R_plugboard', 'R_router',
               'R_stool', 'R_tire', 'R_umbrella', 'R_watch',
               'W_Cabbage', 'W_apple',
               'W_greenVegetables', 'W_hamimelon', 'W_leftovers', 'W_meat', 'W_orange', 'W_pancake', 'W_peel',
               'W_pitahaya', 'W_radish', 'W_strawberry', 'W_tofu', 'W_tomato')

    # 仅加载模型参数
    net = torchvision.models.resnet18(pretrained=False)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 60)

    #net.load_state_dict(torch.load('net_dict.pt'))

    # # 加载完整模型
    net = torch.load('net.pt')

    net.to(DEVICE)
    net.eval()


    # 随机选取一张验证图片
    img_dir = 'E:\\PROJECT\\PycharmProjects\\pt\\test_images\\'
    #img_dir= 'E:\\train_data\\trash_dataset\\new_data\\test_data\\R_desklamp\\'
    test_imgs = os.listdir(img_dir)
    img_file = random.choice(test_imgs)
    img0 = cv2.imread(img_dir+img_file)
    img  = transformer(img0)
    img = img.unsqueeze(0)      # 扩充一个维度，作为输入需要的batchsize维度

    with torch.no_grad():
        outs = net(img.to(DEVICE))
        # print(outs)
        softmax=F.softmax(outs)
        # print(softmax)
        _,max_index_softmax = torch.max(softmax,1)
        #print(max_index_softmax)
        _, pre = torch.max(outs, 1)
        print('the label is{} : {}'.format(pre[0],classes[pre[0]]))      # 输出预测类别

    # 显示图片，按任意键关闭
    cv2.imshow('input_image', img0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
