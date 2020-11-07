#server 코드
from flask import Flask, render_template, jsonify, request
import os, sys, time
# os.chdir(os.path.dirname(__file__))
from PIL import Image,ImageTk
import json,cv2,argparse,torch
import numpy as np
from PIL import Image
from pathlib import Path
from multiprocessing import Process, Pipe,Value,Array
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank
from datetime import datetime
from mtcnn_pytorch.src.align_trans import get_reference_facial_points, warp_and_crop_face
from face_recognition import get_face_feature, get_max_cos #11.16

app = Flask(__name__)

server = "http://127.0.0.1:5000/"
#11.16
register_list=[] 
conf = get_config(False) 
learner = face_learner(conf, True)
learner.threshold = 1.54
if conf.device.type == 'cpu':
    learner.load_state(conf, 'cpu_final.pth', True, True)
else:
    learner.load_state(conf, 'final.pth', True, True)
learner.model.eval()

@app.route('/register',methods=["POST"])
def register():

    print("--------------")

    register_face = request.json['face_image']
    register_np = np.array(register_face)
    register_pil = Image.fromarray(register_np, mode='RGB')
    feature = get_face_feature(conf, learner.model, register_pil)
    register_list.append(feature)
    print(register_list)
    return "register success!"

@app.route('/register_check',methods=["POST"])
def register_check():
    print(">>>>>>>>>>>>>")

    face_list = request.json['face_list']
    check_list = []
    for face in face_list:
        face = np.array(face)
        pil_img = Image.fromarray(face, mode='RGB')
        feature = get_face_feature(conf, learner.model, pil_img)
        cos_sim = get_max_cos(feature, register_list)
        if cos_sim > 0.9:
            check_list.append("known")
        else:
            check_list.append("unknown")
    print(check_list)
    check_list = {'check_list': check_list}
    return jsonify(check_list)

if __name__ =='__main__':
   app.run(host='0.0.0.0', port=5000,debug=True)

