import requests
import argparse
import cv2

# Initialize the PyTorch REST API endpoint URL.
PyTorch_REST_API_URL = 'http://127.0.0.1:5555/predict'


def predict_result(image_path):
    # Initialize image path
    image = open(image_path, 'rb').read()
    payload = {'image': image}

    # Submit the request.两个参数：URL，字典files（键值对{image：二进制数组}）
    r = requests.post(PyTorch_REST_API_URL, files=payload).json()
    print(r['success'])
    # Ensure the request was successful.
    if r['success']:
        print('the result of prediction is: %s'%r['prediction'])
    else:
        print('Request failed')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classification demo')
    parser.add_argument('--file', type=str, help='test image file')
    args = parser.parse_args()
    predict_result(args.file)

    # predict_result('e:/PROJECT/PycharmProjects/pt/test_images/spoon.jpg')