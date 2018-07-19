#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 15:57:14 2018

@author: daphne
"""
import json
import cv2
import copy
import numpy as np
from PIL import Image
from PIL import ImageEnhance
import argparse
import os
from io import StringIO,BytesIO
try: #python3
    from urllib.request import urlopen
except: #python2
    from urllib2 import urlopen


#try to change the output json file path here
ouput_json_path = 'data.txt'

#change minimum box area here
box_area = 180

#change enhance ratio here in case of blur pictures
enhance_ratio = 40.0

#change contour threshold here for the size of line
contour_threshold = 180

#helper function for calculation angle
def angle_cos(p0, p1, p2):
    d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))


  
def draw_image(input_path,output_path):
    if not os.path.exists(output_path):
        print('Not a valid output path,please try again')
        exit()
    try:
        file = BytesIO(urlopen(input_path).read())
        img = Image.open(file)
    except ValueError:
        if os.path.exists(input_path):
            img = Image.open(input_path)
        else:
            print('Not a valid URL or path, please try again')
            exit()

    img_copy = copy.deepcopy(img)
    img = np.asarray(img)
    enhancer = ImageEnhance.Contrast(img_copy)
    img_copy = np.asarray(enhancer.enhance(enhance_ratio))
    
    #convert to gray scale
    gray = cv2.cvtColor(img_copy,cv2.COLOR_BGR2GRAY)
    
    #find contours
    ret,thresh = cv2.threshold(gray[:],contour_threshold,255,cv2.THRESH_BINARY)
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    points = []
    
    #for each contours, do the following
    for cnt in contours:
        cnt_len = cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, 0.02 * cnt_len, True)
        
        #if the shape is conves ond greater than some area, we choose it as a box
        if len(cnt) == 4 and cv2.isContourConvex(cnt) and cv2.contourArea(cnt) > box_area:
            cnt = cnt.reshape(-1, 2)
            max_cos = np.max([angle_cos(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4]) for i in range(4)])
            if max_cos < 0.3:
                rects.append(cnt)
                points.append({'points':cnt.tolist()})
    
    cv2.drawContours(img, rects, -1, (0, 255, 0), 3)
    output_filename = os.path.join(output_path,input_path.split('/')[-1])
    print(output_filename,input_path.split('/')[-1])
    cv2.imwrite(output_filename,cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
    data = {}
    data['boxes'] = points
    with open(ouput_json_path, 'a') as outfile:  
        json.dump(data,outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Draw rectangles in document",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input',type=str, required=True)
    parser.add_argument('-o', '--output',type=str, required=True)
    args = parser.parse_args()
    
    input_path = args.input
    output_path = os.path.abspath(args.output)   
    draw_image(input_path,output_path)