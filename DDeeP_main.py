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
from mtcnn_pytorch.src.align_trans import get_reference_facial_points, warp_and_crop_face

parser = argparse.ArgumentParser(description='take a picture')
parser.add_argument('--name', '-n', default='unknown', type=str, help='input the name of the recording person')
parser.add_argument('-th', '--threshold', help='threshold to decide identical faces', default=1.54, type=float)
parser.add_argument("-u", "--update", help="whether perform update the facebank", action="store_true")

args = parser.parse_args()

conf = get_config(False)
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

server = "http://127.0.0.1:5000/"

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
def ddeep():
    isSuccess, frame = cap.read()
    if isSuccess:
        try:
            image = Image.fromarray(frame)

            bboxes, faces = mtcnn.align_multi(image, conf.face_limit, conf.min_face_size)

            bboxes = bboxes[:, :-1]
            bboxes = bboxes.astype(int)
            bboxes = bboxes + [-1, -1, 1, 1]
            face_list = []

            for idx, bbox in enumerate(bboxes):
                face_list.append(np.array(faces[idx]).tolist())

            URL = server + "register_check"
            json_feed = {'face_list': face_list}
            response = requests.post(URL, json=json_feed)
            check_list = response.json()["check_list"]
            for idx, bbox in enumerate(bboxes):
                if check_list[idx] == 'unknown':
                    frame[bbox[1]: bbox[3], bbox[0]:bbox[2]] = cv2.blur(frame[bbox[1]: bbox[3], bbox[0]: bbox[2]],
                                                                        (23, 23))
                else:
                    frame = draw_box_name(bbox, "known", frame)

            cv2.imshow('DDeeP', frame)
        except:
            print("Sorry ")

        if cv2.waitKey(1) & 0xFF == ord('r'):

            p = Image.fromarray(frame[..., ::-1])
            try:
                register_face = np.array(mtcnn.align(p))[..., ::-1]
                name = 'A'
                URL = server + "register"
                tolist_face = register_face.tolist()
                json_feed = {'register_image': tolist_face, 'register_name': name}

                response = requests.post(URL, json=json_feed)
                print(response)

            except:
                print('no face captured')
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

    #키보드에서 r을 누르면, register가 가능함.


while cap.isOpened():
    ddeep()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break