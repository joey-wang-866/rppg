#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''Code to test FaceAligner2D.

...

Example:
    $ python3 test_facealigner2d.py <videofile>

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
from pyrppg.utils import FaceDetector, FaceAligner2D

# python test_facedetector.py videopath startframe
assert len(sys.argv) > 1, 'Number of inputs should greater than 2'
videopath = sys.argv[1]
startframe = int(sys.argv[2]) if len(sys.argv) > 2 else 0

face_detector = FaceDetector()
face_aligner = FaceAligner2D()
# face_aligner = FaceAligner2D(backend=FaceAligner2D.BACKEND_CV_KAZEMI)
# face_aligner = FaceAligner2D(backend=FaceAligner2D.BACKEND_DLIB_KAZEMI)
cap = cv2.VideoCapture(videopath)
cap.set(cv2.CAP_PROP_POS_FRAMES, startframe)
rotate = cv2.ROTATE_90_CLOCKWISE
cv2.namedWindow('face')
while True:
    ret, img = cap.read()
    if not ret: break
    img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
    img = cv2.rotate(img, rotate)
    bboxes = face_detector.detect(img)
    # print(bboxes)
    if bboxes is not None:
        coords = face_aligner.align(img, bboxes[0])
        for bbox in bboxes:
            cv2.rectangle(img, tuple(bbox[:2]), tuple(bbox[2:]), (0, 0, 255), 2)
        for p in coords:
            cv2.circle(img,tuple(p), 2, (0, 255, 0), 2)
    cv2.imshow('face', img)
    if cv2.waitKey(1) == 27: break
cv2.destroyAllWindows()
