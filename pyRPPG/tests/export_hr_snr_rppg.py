# export heart rate based on different rPPG methods
import sys
sys.path.append('/home/jagmohanmeher/nordlinglab-imageanalysis/pyRPPG')
sys.path.append('/home/jagmohanmeher/nordlinglab-imageanalysis/pyRPPG/pyrppg')
sys.path.append('/home/jagmohanmeher/nordlinglab-imageanalysis/pyRPPG/tests')
from pyrppg.pulse import HYBRID
from pyrppg.utils import FaceDetector, FaceAligner3D, SkinDetector, set_exact_frame
from pyrppg.eval import findspikes, estimate_hr_snr_by_fft
import os
import time
import json
import cv2
import numpy as np
from scipy import signal, fft
sys.path.insert(1, os.path.join(sys.path[0], '..'))

assert len(sys.argv) == 2, 'Usage: python export_bbox_colormap.py xxx.json'

with open(sys.argv[1], 'r', encoding='utf-8') as f:
    exp = json.load(f)
    # for exp1 (list the test you want to process here)
    tests = ['static1', 'static2', 'static3',
             'speak', 'rotate', 'bike1', 'bike2', 'bike3']
    cameras = ['camera1']  # preprocess camera1 only
    cap = cv2.VideoCapture()
    preprocessed_colormap = cv2.VideoCapture()
    bgcolor = (0, 255, 0)
    root = exp['root'] if 'root' in exp else ''
    hybrid = HYBRID()
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
                if 'hybrid' not in subject[test][camera]:
                    continue
                # special cases
                # if subject['id'] == '...' and test == '...' and camera == '...':
                # start
                print(
                    f'----- Subject {subject["id"]} in {test} with {camera} -----')
                t0 = time.time()
                tes = subject[test]
                cam = subject[test][camera]
                # load aligned physio
                try:
                    aligned_physio = np.loadtxt(os.path.join(
                        root, cam['aligned physio']), delimiter=',')
                    istart, iend = int(aligned_physio[0, 0]), int(
                        aligned_physio[-1, 0])+1
                    del aligned_physio
                    # special casesma
                    if subject['id'] == 496 and test == 'bike3':
                        iend = istart + 18098
                except:
                    print('Failed to load aligned physio')
                    continue
                # open video
                if not cap.open(os.path.join(root, cam['video'])):
                    print('Failed to open video')
                    continue
                fs = cap.get(cv2.CAP_PROP_FPS)
                # load preprocessed bbox
                try:
                    preprocessed_bboxes = np.load(
                        os.path.join(root, cam['bbox']))
                except:
                    print('Failed to load preprocessed bbox')
                    continue
                # load preprocessed colormap
                if not preprocessed_colormap.open(os.path.join(root, cam['colormap'])):
                    print('Failed to open preprocessed colormap')
                    continue
                # process each interval
                L, dL = fft.next_fast_len(int(fs * 30)), int(fs * 5)
                hr_green, hr_ica, hr_chrom, hr_pos, hr_sph, hr_csc, hr_chromprn = [], [], [], [], [], [], []
                snr_green, snr_ica, snr_chrom, snr_pos, snr_sph, snr_csc, snr_chromprn = [
                ], [], [], [], [], [], []
                for i in range(0, iend - istart - L + 1, dL):
                    isdark = test == 'static1' or test == 'bike1'
                    s_green, s_ica, s_chrom, s_pos, s_sph, s_csc, s_chromprn = hybrid.extract(
                        cap, i, i+L, offset=istart, rotate=rotate, isdark=isdark,  imputation=True, preprocessed_bboxes=preprocessed_bboxes, preprocessed_colormap=preprocessed_colormap, display=False)
                    hr, snr = estimate_hr_snr_by_fft(s_green, fs)
                    hr_green.append(hr)
                    snr_green.append(snr)
                    hr, snr = estimate_hr_snr_by_fft(s_ica, fs)
                    hr_ica.append(hr)
                    snr_ica.append(snr)
                    hr, snr = estimate_hr_snr_by_fft(s_chrom, fs)
                    hr_chrom.append(hr)
                    snr_chrom.append(snr)
                    hr, snr = estimate_hr_snr_by_fft(s_pos, fs)
                    hr_pos.append(hr)
                    snr_pos.append(snr)
                    hr, snr = estimate_hr_snr_by_fft(s_sph, fs)
                    hr_sph.append(hr)
                    snr_sph.append(snr)
                    hr, snr = estimate_hr_snr_by_fft(s_csc, fs)
                    hr_csc.append(hr)
                    snr_csc.append(snr)
                    hr, snr = estimate_hr_snr_by_fft(s_chromprn, fs)
                    hr_chromprn.append(hr)
                    snr_chromprn.append(snr)
                    print(hr*60)
                # release
                cap.release()
                preprocessed_colormap.release()
                del preprocessed_bboxes
                # save
                hr_green = np.array(hr_green, dtype=np.float64)
                snr_green = np.array(snr_green, dtype=np.float64)
                hr_ica = np.array(hr_ica, dtype=np.float64)
                snr_ica = np.array(snr_ica, dtype=np.float64)
                hr_chrom = np.array(hr_chrom, dtype=np.float64)
                snr_chrom = np.array(snr_chrom, dtype=np.float64)
                hr_pos = np.array(hr_pos, dtype=np.float64)
                snr_pos = np.array(snr_pos, dtype=np.float64)
                hr_sph = np.array(hr_sph, dtype=np.float64)
                snr_sph = np.array(snr_sph, dtype=np.float64)
                hr_csc = np.array(hr_csc, dtype=np.float64)
                snr_csc = np.array(snr_csc, dtype=np.float64)
                hr_chromprn = np.array(hr_chromprn, dtype=np.float64)
                snr_chromprm = np.array(snr_chromprn, dtype=np.float64)
                np.savez(os.path.join(root, cam['hybrid']),
                         hr_green=hr_green, snr_green=snr_green,
                         hr_ica=hr_ica, snr_ica=snr_ica,
                         hr_chrom=hr_chrom, snr_chrom=snr_chrom,
                         hr_pos=hr_pos, snr_pos=snr_pos,
                         hr_sph=hr_sph, snr_sph=snr_sph,
                         hr_csc=hr_csc, snr_csc=snr_csc,
                         hr_chromprn=hr_chromprn, snr_chromprn=snr_chromprn)
                print(f'Time cost: {time.time()-t0:.3f} seconds')
