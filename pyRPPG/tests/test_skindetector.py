#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''Code to test SkinDetector.

...

Example:
    $ python3 test_skindetector.py <videofile>

Todo:
    * ...
    * ...
'''
import sys
sys.path.append('/home/jagmohanmeher/nordlinglab-imageanalysis/pyRPPG')
sys.path.append('/home/jagmohanmeher/nordlinglab-imageanalysis/pyRPPG/pyrppg')
sys.path.append('/home/jagmohanmeher/nordlinglab-imageanalysis/pyRPPG/tests')
import os

import cv2
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from pyrppg.utils import SkinDetector

# python test_facedetector.py videopath startframe
assert len(sys.argv) > 1, 'Number of inputs should greater than 2'
videopath = sys.argv[1]
startframe = int(sys.argv[2]) if len(sys.argv) > 2 else 0

skin_detector = SkinDetector()
# skin_detector = SkinDetector(backend=SkinDetector.BACKEND_YCRCB)
# skin_detector = SkinDetector(backend=SkinDetector.BACKEND_RCA)
# skin_detector = SkinDetector(backend=SkinDetector.BACKEND_XYZ)
# skin_detector = SkinDetector(backend=SkinDetector.BACKEND_HLS)
cap = cv2.VideoCapture(videopath)
cap.set(cv2.CAP_PROP_POS_FRAMES, startframe)
rotate = cv2.ROTATE_90_CLOCKWISE
cv2.namedWindow('img')
while True:
    ret, img = cap.read()
    if not ret: break
    img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
    img = cv2.rotate(img, rotate)
    mask = skin_detector.detect(img)
    cv2.imshow('img', img)
    cv2.imshow('mask', mask)
    if cv2.waitKey(1) == 27: break
cv2.destroyAllWindows()
