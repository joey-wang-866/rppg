#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''Code to test set_exact_frame.

...

Example:
    $ python3 test_set_exact_frame.py <videofile>

Todo:
    * ...
    * ...
'''
import sys
sys.path.append('/home/jagmohanmeher/nordlinglab-imageanalysis/pyRPPG')
sys.path.append('/home/jagmohanmeher/nordlinglab-imageanalysis/pyRPPG/pyrppg')
sys.path.append('/home/jagmohanmeher/nordlinglab-imageanalysis/pyRPPG/tests')
import os
import time
import cv2
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from pyrppg.utils import set_exact_frame

# python test.py videopath startframe
assert len(sys.argv) > 1, 'Number of inputs should greater than 2'
videopath = sys.argv[1]
startframe = int(sys.argv[2]) if len(sys.argv) > 2 else 0

cap = cv2.VideoCapture(videopath)

t = time.time()
set_exact_frame(cap, 10)
print(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
print(time.time() - t)

t = time.time()
set_exact_frame(cap, 111)
print(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
print(time.time() - t)

t = time.time()
set_exact_frame(cap, 0)
print(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
print(time.time() - t)

t = time.time()
set_exact_frame(cap, 909)
print(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
print(time.time() - t)