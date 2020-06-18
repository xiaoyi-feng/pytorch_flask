import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
import time
import copy


BATCH_SIZE = 100      # 每一批图片数量
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')     # 使用GPU

# 转换为Tensor数据并归一化
transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.299, 0.224, 0.255])
])

# 数据加载器,DataLoader将输入的数据按照batch_size封装成tensor
# 然后，后续需要再包装成Variable就可以作为模型的输入 了
train_dataset = datasets.ImageFolder(root='E:/train_data/trash_dataset/new_data/train_data/', transform=transformer)
test_dataset = datasets.ImageFolder(root='E:/train_data/trash_dataset/new_data/test_data/', transform=transformer)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


'''
一个通用的训练函数，返回训练过程中表现最优的模型
'''
def train(model, criterion, optimizer, epochs):
    since = time.time()

    best_acc = 0.0      # 记录模型测试时的最高准确率
    best_model_wts = copy.deepcopy(model.state_dict())  # 记录模型测试出的最佳参数

    for epoch in range(epochs):
        print('-' * 30)
        print('Epoch {}/{}'.format(epoch+1, epochs))

        # 训练模型
        net.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # 前向传播，计算损失
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # 反向传播+优化
            optimizer.zero_grad()
            loss.backward()
            # 进行梯度更新
            optimizer.step()
            #scheduler.step()

            running_loss += loss.item()

            # 每x批图片打印训练数据
            if (i != 0) and (i % 100 == 0):
                print('step: {:d},  loss: {:.3f}'.format(i, running_loss/100))
                running_loss = 0.0

        # 每个epoch以测试集数据的整体准确率为标准，测试一下模型
        net.eval()
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
        acc = correct / total
        if acc > best_acc:      # 当前准确率更高时更新
            best_acc = acc
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('-' * 30)
    print('训练用时： {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
    print('最高准确率: {}%'.format(100 * best_acc))

    # 返回测试出的最佳模型
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    # 加载预训练的resnet18模型
    net = torchvision.models.resnet18(pretrained=True)
    # 冻结原网络参数，仅训练最后新替换的全连接层
    for param in net.parameters():
        param.requires_grad = False

    num_ftrs = net.fc.in_features   # 原网络最后一层的输入维度
    net.fc = nn.Linear(num_ftrs, 60) # 替换新的连接层，输出改为60，预测60个类别

    net = net.to(DEVICE)

    criterion = nn.CrossEntropyLoss()   # 交叉熵做损失函数
    # SGD优化器，epoch=5时loss已基本不会减小了（0.37~0.46）
    # optimizer = optim.SGD(net.fc.parameters(), lr=0.001, momentum=0.9)
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=800, gamma=0.1) # 学习率调整

    # 将优化器换成Adam试试
    optimizer = optim.Adam(net.fc.parameters(), lr = 0.001)


    # net是train过程中准确度最高的模型（包好训练好的参数）
    net = train(net, criterion, optimizer, 70) # epoch=70

    # 保存模型参数net.state_dict()
    torch.save(net.state_dict(), 'net_dict.pt')
    print('保存模型参数完成')

    # 保存完整模型
    torch.save(net, 'net.pt')
    print('保存模型完成')
