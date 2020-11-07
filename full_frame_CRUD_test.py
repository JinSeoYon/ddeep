import cv2
from PIL import Image
import argparse
from pathlib import Path
from multiprocessing import Process, Pipe,Value,Array
import torch
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank
import sys
import requests
from face_recognition import face_compare
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-s", "--save", help="whether save",action="store_true")
    parser.add_argument('-th','--threshold',help='threshold to decide identical faces',default=1.54, type=float)
    parser.add_argument("-u", "--update", help="whether perform update the facebank",action="store_true")
    parser.add_argument("-tta", "--tta", help="whether test time augmentation",action="store_true")
    parser.add_argument("-c", "--score", help="whether show the confidence score",action="store_true")
    args = parser.parse_args()

    conf = get_config(False)
    server = "http://127.0.0.1:5000/"
    
    mtcnn = MTCNN()
    print('mtcnn loaded')

    # inital camera
    img = []
    
    img.append(cv2.imread('data/input/evans/evans_p.jpg'))
    img.append(cv2.imread('data/input/hermsworth/hermsworth_p.jpg'))
    img.append(cv2.imread('data/input/jeremy/jeremy.jpg'))
    img.append(cv2.imread('data/input/mark/mark.jpg'))
    img.append(cv2.imread('data/input/olsen/olsen.jpg'))
    
    register_face = img[0].tolist()
    name = 'evans'
    URL = server + "register"
    json_feed = {'register_image': register_face, 'register_name': name}
    response = requests.post(URL, json=json_feed)
    print(response.text)
    
    URL = server + "ReadFeature"
    params = {'name': 'evans'}
    res = requests.get(URL, params=params)
    res = res.json()
    res = res['result']
    print(res)
    
    URL = server + 'update'
    params = {'old_name': 'evans', 'new_name': 'jeremy'}
    res = requests.get(URL, params=params)
    print(res.text)
    
    URL = server + 'update'
    new_img = img[2].tolist()
    json_feed = {'name': 'jeremy', 'new_image': new_img}
    res = requests.post(URL, json=json_feed)
    print(res.text)
    
    URL = server + 'delete'
    params = {'name' : 'jeremy'}
    res = requests.delete(URL, params=params)
    print(res.text)
    
    blur_img = img[1].tolist()
    URL = server + "register_check"
    json_feed = {'full_frame': blur_img}
    response = requests.post(URL, json=json_feed)
    res = response.json()
    res = res['check_img']
    res = np.array(res)
    res = np.uint8(res)
    cv2.imshow('image', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
