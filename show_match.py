import cv2
import os
import numpy as np

val_path = '/media/chen/6f586f18-792a-40fd-ada6-59702fb5dabc/data/compe/det/val/images/'
f = open('match_res', 'r')
for line in f.readlines():
    print(line.strip('\n').split('\t'))
    vedio_path, img_path = line.strip('\n').split('\t')
    print(val_path+vedio_path)
    vedio_img = cv2.imread(val_path+vedio_path)
    match_img = cv2.imread(val_path+img_path)
    vedio_img = cv2.resize(vedio_img, (512,980))
    match_img = cv2.resize(match_img, (512, 980))
    img = np.concatenate((vedio_img, match_img), axis=1)
    cv2.imshow('img', img)
    cv2.waitKey(0)
