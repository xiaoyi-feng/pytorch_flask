import io
import json
import cv2
import flask
import numpy as np
import torch
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from flask import request
from torch import nn
from torchvision import transforms as T
from flask import Flask
from flask_sqlalchemy import SQLAlchemy


from torchvision.models import resnet18
# Initialize our Flask application and the PyTorch model.
app = flask.Flask(__name__)

model = None
use_gpu = True

# with open('resnet18_class.txt', 'r',encoding='UTF-8') as f:
#     idx2label = eval(f.read())
#
# print(idx2label)

def load_model():
    """Load the pre-trained model, you can use your model just as easily.
    """
    global model
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 使用GPU
    # 仅加载模型参数
    model = torchvision.models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 60)
    model.load_state_dict(torch.load('E:/PROJECT/PycharmProjects/pt/net_dict.pt'))

    model.to(DEVICE)
    model.eval()


def prepare_image(image, target_size):
    """Do image preprocessing before prediction on any data.
    :param image:       original image
    :param target_size: target image size
    :return:
                        preprocessed image
    """

    if image.mode != 'RGB':
        image = image.convert("RGB")

    # Resize the input image nad preprocess it.
    image = T.Resize(target_size)(image)
    image = T.ToTensor()(image)

    # Convert to Torch.Tensor and normalize.
    image = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)

    # Add batch_size axis.
    image = image[None]
    if use_gpu:
        image = image.cuda()
    return torch.autograd.Variable(image, volatile=True)


@app.route("/predict", methods=["GET","POST"])
def predict():
    with open('resnet18_class.txt', 'r', encoding='UTF-8') as f:
        idx2label = eval(f.read())

    # Initialize the data dictionary that will be returned from the view.
    data = {'success': False}
    # Ensure an image was properly uploaded to our endpoint.
    if flask.request.method == 'POST':
        print("the request method is post!")
        if flask.request.files.get('image'): # 程序没有进入到这个分支里面
        #if flask.request.form['image']:
            print(data['success'])
            # Read the image in PIL format
            # 网络传输中一般选择二进制传输，此步获得request请求中的image数据，为binary格式
            image = flask.request.files['image'].read()
            # 将二进制转化为<PIL.PngImagePlugin.PngImageFile>对象
            image = Image.open(io.BytesIO(image))

            # Preprocess the image and prepare it for classification.
            image = prepare_image(image, target_size=(224, 224))
            # Classify the input image and then initialize the list of predictions to return to the client.
            outs=model(image)
            _, pre = torch.max(outs, 1)
            print(pre)  # tensor([0], device='cuda:0')
            print(int(pre))
            label_name = idx2label[int(pre)]

            # data['prediction']= label_name
            data={'prediction':label_name}
            print(data['prediction'])

            # preds = F.softmax(model(image), dim=1)
            # results = torch.topk(preds.cpu().data, k=3, dim=1)
            # data['predictions'] = list()

            # Indicate that the request was a success.
            # data['success'] = True
            # Return the data dictionary as a JSON response.
    f.close()
    return flask.jsonify(data)

@app.route('/')
def test_demo():
    return '服务器正常运行'

@app.route("/search", methods=["GET","POST"])
def search():
    # 必须将数组或字典转换成json后再返回
    data_list = {}   # 用于存储查询的结果
    data = json.loads(request.get_data())  # 获取前端POST请求传过来的 json 数据

    #key, value = data.items() # dict_items([('wastename', '鼠标')])
    # 遍历字典列表
    for key,value in data.items():
        print(value)
    with open('wasteInfo.txt', 'r', encoding='UTF-8') as f:
        line = f.readline()
        i = 1
        while line:
            if value in line:
                data_list[str(i)] = line
                i+=1
            line = f.readline()
    f.close()
    print(data_list)
    return flask.jsonify(data_list)




# @app.route("/")
# def hello_world():
#     return 'hello,world!'

if __name__ == '__main__':
    print("Loading PyTorch model and Flask starting server ...")
    print("Please wait until server has fully started")
    load_model()
    app.run(host='192.168.43.190',port=5555)