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
from face_recognition import face_compare
import requests
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
    
    faces=[]
    re_img = Image.fromarray(img[0][...,::-1])
    re_img = mtcnn.align(re_img)
    tolist_face = np.array(re_img).tolist()
    URL = server + "register"
    json_feed = {'face_image': tolist_face}
    response = requests.post(URL, json=json_feed)
    print(response)
    
    tolist_face = img[1].tolist()
    URL = server + "getframe"
    json_feed = {'face_list': tolist_face}
    response = requests.post(URL, json=json_feed)
 
    '''   
    for i in range(5):
    
        image = Image.fromarray(img[i][...,::-1]) #bgr to rgb
    #            image = Image.fromarray(img[i])
        #image = mtcnn.align(image)
        #print(image)
        tolist_face = np.array(image).tolist()
        faces.append(tolist_face) 
            
           
    URL = server + "getframe"
    json_feed = {'face_list': faces}
    response = requests.post(URL, json=json_feed)
    
    print(response)
    
    '''      
