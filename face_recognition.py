from torchvision import transforms as trans
import numpy as np
from numpy.linalg import norm
from model import l2_norm
from torch.nn.functional import cosine_similarity
import torch
from PIL import Image

def face_compare(conf, model, faces, target_embs, tta=False):
    embs = []
    results = []
    scores = []
    for img in faces:
        if tta:
            mirror = trans.functional.hflip(img)
            emb = model(conf.test_transform(img).to(conf.device).unsqueeze(0))
            emb_mirror = model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
            embs.append(l2_norm(emb + emb_mirror))
        else:
            embs.append(model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
    for em in embs:
        result, score = check_features(em, target_embs)

        results.append(result)
        scores.append(score)
    
    return results, score
    
def cos_sim(A, B):
    return np.dot(A, B) / (norm(A) * norm(B))


def check_features(emb, target_embs):
    max_idx = 0
    max_sim = 0
    for i in range(target_embs.shape[0]):
        sim = cosine_similarity(emb, target_embs[i].unsqueeze(0))
        if(sim>max_sim):
            max_idx = i
            max_sim = sim
    return max_idx, max_sim


def get_face_feature(conf, model, img, tta=False):
    if tta:
        mirror = trans.functional.hflip(img)
        emb = model(conf.test_transform(img).to(conf.device).unsqueeze(0))
        emb_mirror = model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
        feature = l2_norm(emb + emb_mirror)
    else:
        feature = model(conf.test_transform(img).to(conf.device).unsqueeze(0))
        #feature = model(conf.test_transform(img).to(conf.device))

    return feature

def get_max_cos(feature, feature_list):
    max_sim = 0
    for compare in feature_list:
        sim = cosine_similarity(feature, compare)
        if(sim>max_sim):
            max_sim = sim
    return max_sim