# export preprocessed bbox and colormap (start from istart (trigger) in aligned physio)
import sys
sys.path.append('/home/jagmohanmeher/nordlinglab-imageanalysis/pyRPPG')
sys.path.append('/home/jagmohanmeher/nordlinglab-imageanalysis/pyRPPG/pyrppg')
sys.path.append('/home/jagmohanmeher/nordlinglab-imageanalysis/pyRPPG/tests')
from pyrppg.utils import FaceDetector, FaceAligner3D, set_exact_frame
import os
import time
import json
import cv2
import numpy as np
sys.path.insert(1, os.path.join(sys.path[0], '..'))

assert len(sys.argv) == 2, 'Usage: python export_bbox_colormap.py list_xxx.json'

with open(sys.argv[1], 'r', encoding='utf-8') as f:
    exp = json.load(f)
    # for exp1 (list the test you want to process here)
    tests = ['static1', 'static2', 'static3',
             'speak', 'rotate', 'bike1', 'bike2', 'bike3']
    cameras = ['camera1']  # preprocess camera1 only
    rotates = [cv2.ROTATE_90_CLOCKWISE]
    cap = cv2.VideoCapture()
    wri = cv2.VideoWriter()
    wri_args = dict(
        apiPreference=cv2.CAP_FFMPEG,  # use FFMEPG backend
        fourcc=cv2.VideoWriter_fourcc(*'FFV1'),  # lossless compression
        frameSize=(512, 512)
    )
    bgcolor = (0, 255, 0)
    empty_colormap = np.full((512, 512, 3), bgcolor, dtype=np.uint8)
    face_detector = FaceDetector()
    face_aligner = FaceAligner3D()
    root = exp['root'] if 'root' in exp else ''
    for subject in exp['subjects']:
        # different rotation angle for some videos
        if subject in [660, 984, 764, 301]:
            rotates = [cv2.ROTATE_90_COUNTERCLOCKWISE]
        else:
            rotates = [cv2.ROTATE_90_CLOCKWISE]

        for test in tests:
            for camera, rotate in zip(cameras, rotates):
                # check fields
                if 'id' not in subject:
                    continue
                if test not in subject:
                    continue
                if camera not in subject[test]:
                    continue
                if 'video' not in subject[test][camera]:
                    continue
                if 'bbox' not in subject[test][camera]:
                    continue
                if 'colormap' not in subject[test][camera]:
                    continue
                if 'aligned physio' not in subject[test][camera]:
                    continue
                if 'gt' not in subject[test][camera]:
                    continue
                if 'physio' not in subject[test]:
                    continue
                if 'ecg avf' not in subject[test]['physio']:
                    continue
                # TODO: check existence of files
                # start
                print(
                    f'----- Subject {subject["id"]} in {test} with {camera} -----')
                t0 = time.time()
                tes = subject[test]
                cam = subject[test][camera]
                # load aligned physio (requires the index of triggered frame (the first reading))
                try:
                    aligned_physio = np.loadtxt(os.path.join(
                        root, cam['aligned physio']), delimiter=',')
                    istart, iend = int(aligned_physio[0, 0]), int(
                        aligned_physio[-1, 0])+1
                    del aligned_physio
                    # iend = istart + 50 # for test
                except:
                    print('Failed to load aligned physio')
                    continue
                # open video
                if not cap.open(os.path.join(root, cam['video'])):
                    print('Failed to open video')
                    continue
                # open writer for colormap preprocessed_colormap
                if not wri.open(os.path.join(root, cam['colormap']), fps=cap.get(cv2.CAP_PROP_FPS), **wri_args):
                    print('Failed to open colormap')
                    cap.release()
                    continue
                # open preprocessed_bboxes, and start to process
                with open(os.path.join(root, cam['bbox']), 'wb') as fbbox:
                    set_exact_frame(cap, istart)
                    bbox = np.full((iend - istart, 5),
                                   (-1, 0, 0, 0, 0), dtype=np.int32)
                    for i in range(iend - istart):
                        ret, img = cap.read()
                        if not ret:
                            break
                        if rotate is not None:
                            img = cv2.rotate(img, rotate)
                        # face detection
                        bboxes = face_detector.detect(img)
                        if bboxes is not None:
                            x1, y1, x2, y2 = tuple(bboxes[0])
                            bbox[i] = np.array(
                                [1, x1, y1, x2, y2], dtype=np.int32)
                            # face alignment
                            colormap = face_aligner.align(
                                img, bboxes[0], bgcolor=bgcolor)
                            wri.write(colormap)
                        else:
                            wri.write(empty_colormap)
                        if i % 1000 == 999:
                            print(f'Processed {i+1} frames')
                    np.save(fbbox, bbox)
                # release
                wri.release()
                cap.release()
                print(f'Time cost: {time.time()-t0:.3f} seconds')
