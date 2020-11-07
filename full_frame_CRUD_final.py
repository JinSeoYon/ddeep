import cv2
from PIL import Image
import argparse
from pathlib import Path
from multiprocessing import Process, Pipe, Value, Array
import torch
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank
import sys
import requests
from face_recognition import face_compare
import numpy as np


parser = argparse.ArgumentParser(description='for face verification')
parser.add_argument("-s", "--save", help="whether save", action="store_true")
parser.add_argument('-th', '--threshold', help='threshold to decide identical faces', default=1.54, type=float)
parser.add_argument("-u", "--update", help="whether perform update the facebank", action="store_true")
parser.add_argument("-tta", "--tta", help="whether test time augmentation", action="store_true")
parser.add_argument("-c", "--score", help="whether show the confidence score", action="store_true")
args = parser.parse_args()

conf = get_config(False)
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

server = "http://127.0.0.1:5000/"

mtcnn = MTCNN()
print('mtcnn loaded')

def ddeep():
    isSuccess, frame = cap.read()
    if isSuccess:
        try:
            print("wait for loading ... ")
            image = Image.fromarray(frame)
            cv2.imshow('DDeeP',frame)
            """
            frame_to_server = np.array(image).tolist()
            
            URL = server + "register_check"
            json_feed = {'full_frame': frame_to_server}
            response = requests.post(URL, json=json_feed)
            res = res['check_img']
            res = np.array(res)
            res = np.uint8(res)
            cv2.imshow('image', res)
            """
            if cv2.waitKey(1) & 0xFF == ord('r'):
                takepic = Image.fromarray(frame[..., ::-1])
                try:
                    register_face = np.array(takepic).tolist()
                    name = 'A'
                    URL = server + "register"
                    json_feed = {'register_image': register_face, 'register_name': name}
                    response = requests.post(URL, json=json_feed)
                    print(response.text)
                except:
                    print('error in register process')
            # 키보드에서 c를 누르면 confirm
            if cv2.waitKey(0) & 0xFF == ord('c'):
                URL = server + "ReadFeature"
                params = {'name': 'A'}
                res = requests.get(URL, params=params)
                res = res.json()
                res = res['result']
                print(res)
            # 키보드에서 n를 누르면 name update
            if cv2.waitKey(0) & 0xFF == ord('n'):
                URL = server + 'update'
                params = {'old_name': 'A', 'new_name': 'NEW'}
                res = requests.get(URL, params=params)
                print(res.text)
            # 키보드에서 u를 누르면 등록된 얼굴을 update할 수 있다.
            if cv2.waitKey(0) & 0xFF == ord('u'):
                newpic = Image.fromarray(frame[..., ::-1])
                new_img = np.array(newpic).tolist()
                URL = server + 'update'
                json_feed = {'name': 'NEW', 'new_image': new_img}
                res = requests.post(URL, json=json_feed)
            # 키보드에서 d를 누르면 삭제가능.
            if cv2.waitKey(0) & 0xFF == ord('d'):
                URL = server + 'delete'
                params = {'name': 'NEW'}
                res = requests.delete(URL, params=params)
                print(res.text)
        except:
            print("detect error")
    #키보드에서 r을 누르면, register가 가능함.


while cap.isOpened():
    ddeep()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break