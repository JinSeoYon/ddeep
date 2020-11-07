#진서연 server test 코드 얼굴 등록후 얼굴의 피쳐값 전달.
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

args = parser.parse_args()
from pathlib import Path

data_path = Path('data')
save_path = data_path / 'facebank' / args.name
if not save_path.exists():
    save_path.mkdir()

cap = cv2.VideoCapture(0)

cap.set(3,1280)
cap.set(4,720)
mtcnn = MTCNN()

def get_pic():
    isSuccess, frame = cap.read()

    if isSuccess:
        frame_text = cv2.putText(frame,
                                 'Press t to take a picture,q to learn.....',
                                 (10, 100),
                                 cv2.FONT_HERSHEY_SIMPLEX,
                                 2,
                                 (0, 255, 0),
                                 3,
                                 cv2.LINE_AA)
        cv2.imshow("My Capture", frame_text)

    if cv2.waitKey(1) & 0xFF == ord('t'):
        p = Image.fromarray(frame[..., ::-1])

        try:
            warped_face = np.array(mtcnn.align(p))[..., ::-1]

            re_img = mtcnn.align(p)
            tolist_face = np.array(re_img).tolist()

            cv2.imwrite(str(save_path / '{}.jpg'.format(str(datetime.now())[:-7].replace(":", "-").replace(" ", "-"))),
                        warped_face)
            URL = server + "register"

            json_feed = {'face_image': tolist_face}
            response = requests.post(URL, json=json_feed)
            print(response)

        except:
            print('no face captured')

while cap.isOpened():
    get_pic()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break













