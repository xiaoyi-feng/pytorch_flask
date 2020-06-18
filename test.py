import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
import cv2
import os

if __name__ == '__main__':

    BATCH_SIZE = 60
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')     # 使用GPU


    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.299, 0.224, 0.255])
    ])
    test_dataset = datasets.ImageFolder(root='E:\\train_data\\trash_dataset\\new_data\\test_data\\', transform=transformer)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # classes = ('cardboard','glass','metal','organism','paper','plastic')
    classes = ('D_adhesiveTape', 'D_cottonswabs', 'D_faceMask', 'D_featherDuster', 'D_flyswatter',
               'D_foodBox','D_glassesCloth','D_makerPen','D_napkin','D_strawHat', 'D_toothBrush','D_towel',
                'H_battery', 'H_drycell', 'H_lamp', 'H_pesticide','H_pills', 'H_solarPanel','H_thermometer',
               'R_TV', 'R_TrolleyCase','R_bag', 'R_bedding', 'R_calculator','R_cardboard','R_cellPhone',
               'R_chargeLine', 'R_desk', 'R_desklamp', 'R_earphone', 'R_fanner', 'R_fireExtinguisher',
               'R_glass', 'R_keyboard','R_magicMouse','R_magneticCard',
               'R_metal', 'R_microphone', 'R_paper', 'R_plastic', 'R_plugboard', 'R_router',
               'R_stool', 'R_tire','R_umbrella', 'R_watch',
               'W_Cabbage', 'W_apple',
               'W_greenVegetables', 'W_hamimelon', 'W_leftovers', 'W_meat', 'W_orange', 'W_pancake', 'W_peel',
               'W_pitahaya', 'W_radish', 'W_strawberry', 'W_tofu', 'W_tomato')



    # 仅加载模型参数
    net = torchvision.models.resnet18(pretrained=False)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 60)
    net.load_state_dict(torch.load('net_dict.pt'))

    # # 加载完整模型
    # net = torch.load('net.pt')

    net.to(DEVICE)
    #  #net.eval(): 在运行推断之前，将dropout和batch规范化层设置为评估模式。
    #  如果不这样做，将会产生不一致的推断结果。
    net.eval()

    # 整体正确率
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('整体准确率: {}%'.format(100 * correct / total))

        print('=====================================================')

        # 每一个类别的正确率
        class_correct = list(0 for i in range(len(classes)))
        class_total = list(0 for i in range(len(classes)))
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                #print(labels)
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(len(classes)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        for i in range(len(classes)):
            print('{}的准确率 : {:.2f}%'.format(classes[i], 100 * class_correct[i] / class_total[i]))
