
# server 코드

from PIL import Image
import argparse
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, prepare_facebank
import cv2
import numpy as np

from flask import Flask, render_template, jsonify, request
from mtcnn_pytorch.src.align_trans import get_reference_facial_points, warp_and_crop_face
from face_recognition import get_face_feature, get_max_cos  # 11.16

app = Flask(__name__)

server = "http://127.0.0.1:5000/"
# 11.16
parser = argparse.ArgumentParser(description='for face verification')
parser.add_argument("-s", "--save", help="whether save", action="store_true")
parser.add_argument('-th', '--threshold', help='threshold to decide identical faces', default=1.54, type=float)
parser.add_argument("-u", "--update", help="whether perform update the facebank", action="store_true")
parser.add_argument("-tta", "--tta", help="whether test time augmentation", action="store_true")
parser.add_argument("-c", "--score", help="whether show the confidence score", action="store_true")
args = parser.parse_args()
conf = get_config(False)

name_list = []
register_list = []

mtcnn = MTCNN()
print("mtcnn loaded")
learner = face_learner(conf, True)

learner.threshold = args.threshold

learner.model.eval()
print('learner loaded')


@app.route('/register',methods=["POST"])
def register():
    print("--------------")

    register_face = request.json['register_image']
    register_np = np.array(register_face)
    register_np = np.uint8(register_np)
    register_pil = Image.fromarray(register_np)
    
    register_name = request.json['register_name']
   
    bbox, face = mtcnn.align(register_pil, conf.min_face_size)
    feature = get_face_feature(conf, learner.model, face)
    name_list.append(register_name)
    register_list.append(feature)
    
    return register_name + " register success!"

@app.route('/ReadFeature')
def ReadFeature():
    ReadName = request.args.get('name')
    try:
        result_idx = name_list.index(ReadName)
        result = register_list[result_idx]
        result = result.tolist()
    except:
        result = ReadName + ' is not a registered face'
    
    read = {'result' : result}
    
    return jsonify(read)
      
@app.route('/update', methods=["POST", "GET"])
def update():
    if request.method == 'GET':
        old_name = request.args.get('old_name')
        new_name = request.args.get('new_name')
        try:
            name_idx = name_list.index(old_name)
            name_list[name_idx] = new_name
            return new_name + ' updated'
        except:
            return old_name + ' is not a registered face'
    else:
        name = request.json['name']
        new_img = request.json['new_image']
        try:
            idx = name_list.index(name)
            new_np = np.array(new_img)
            new_np = np.uint8(new_np)
            new_pil = Image.fromarray(new_np)
            bbox, face = mtcnn.align(new_pil, conf.min_face_size)
            feature = get_face_feature(conf, learner.model, face)
            register_list[idx] = feature
            return name + ' image updated'
        except:
            return name + ' is not a registered face'

    
@app.route('/delete', methods=["DELETE"])
def delete():
    name = request.args.get('name')
    try:
        idx = name_list.index(name)
        del name_list[idx]
        del register_list[idx]
        return name + ' is deleted'
    except:
        return name + ' is not a registered face'

#frame받아서 모자이크한 이미지 보내기
@app.route('/register_check',methods=["POST"])
def register_check():
    print(">>>>>>>>>>>>>")
    check_img = request.json['full_frame']
    check_np = np.array(check_img)
    check_np = np.uint8(check_np)
    check_pil = Image.fromarray(check_np)
    
    bboxes, faces = mtcnn.align_multi(check_pil, conf.face_limit, conf.min_face_size)
    bboxes = bboxes[:, :-1]  # shape:[10,4],only keep 10 highest possibiity faces
    bboxes = bboxes.astype(int)
    bboxes = bboxes + [-1, -1, 1, 1]
    
    for idx,bbox in enumerate(bboxes):
        feature = get_face_feature(conf, learner.model, faces[idx])
        cos_sim = get_max_cos(feature, register_list)

        if cos_sim < 0.9:
            check_np[bbox[1] : bbox[3], bbox[0] : bbox[2]] = cv2.blur(check_np[bbox[1] : bbox[3], bbox[0] : bbox[2]], (23,23))
    tolist_img = check_np.tolist()
    
    check_img = {'check_img': tolist_img}
    return jsonify(check_img)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

