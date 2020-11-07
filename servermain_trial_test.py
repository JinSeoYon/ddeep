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
import requests
import os,sys,time
import numpy as np
from datetime import datetime

from mtcnn_pytorch.src.align_trans import get_reference_facial_points, warp_and_crop_face
#from face_verify_module import fn_face_verify
#from take_pic_module import get_pic


server = "http://127.0.0.1:5000/"

parser = argparse.ArgumentParser(description='take a picture')
parser.add_argument('--name', '-n', default='unknown', type=str, help='input the name of the recording person')
parser.add_argument('-th', '--threshold', help='threshold to decide identical faces', default=1.54, type=float)
parser.add_argument("-u", "--update", help="whether perform update the facebank", action="store_true")

args = parser.parse_args()

conf = get_config(False)

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

mtcnn = MTCNN()
print('mtcnn loaded')
learner = face_learner(conf, True)
learner.threshold = args.threshold
if conf.device.type == 'cpu':
    learner.load_state(conf, 'cpu_final.pth', True, True)
else:
    learner.load_state(conf, 'final.pth', True, True)
learner.model.eval()
print('learner loaded')

if args.update:
    targets, names = prepare_facebank(conf, learner.model, mtcnn, tta=args.tta)
    print('facebank updated')
else:
    targets, names = load_facebank(conf)
    print('facebank loaded')

def get_pic():
    isSuccess, frame = cap.read()

    if isSuccess:
        try:
            image = Image.fromarray(frame)
            bboxes, faces = mtcnn.align_multi(image, conf.face_limit, conf.min_face_size)
            bboxes = bboxes[:, :-1]  # shape:[10,4],only keep 10 highest possibiity faces
            bboxes = bboxes.astype(int)
            bboxes = bboxes + [-1, -1, 1, 1]  # personal choice
            face_list = []
            for idx, bbox in enumerate(bboxes):
                face_list.append(np.array(faces[idx]).tolist())

            URL = server + "register_check"
            json_feed_verify = {'face_list': face_list}
            start_time = datetime.now()
            response = requests.post(URL, json=json_feed_verify)
            finish_time = datetime.now()
            print('register check time:', finish_time - start_time)
            print(response)
            check_list = response.json()["check_list"]
            for idx, bbox in enumerate(bboxes):
                if check_list[idx] == 'unknown':
                    frame[bbox[1] : bbox[3], bbox[0] : bbox[2]] = cv2.blur(frame[bbox[1] : bbox[3], bbox[0] : bbox[2]], (23,23))
                else:
                    frame = draw_box_name(bbox,"known", frame)
            
            
            
            cv2.imshow("My Capture", frame)
        except:
            print("detect error")

    if cv2.waitKey(1) & 0xFF == ord('t'):
        p = Image.fromarray(frame[..., ::-1])
        try:
            warped_face = np.array(mtcnn.align(p))[..., ::-1]
            re_img = mtcnn.align(p)
            tolist_face = np.array(re_img).tolist()
            URL = server + "register"
            tolist_img = warped_face.tolist()
            json_feed = {'face_image': tolist_face}
            sregister = datetime.now()
            response = requests.post(URL, json=json_feed)
            fregister = datetime.now()

        except:
            print('no face captured')

while cap.isOpened():
    get_pic()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break













