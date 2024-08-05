# export ground truth heart rate by ECG (gtp) and PPG (gt)

import sys
sys.path.append('/home/jagmohanmeher/nordlinglab-imageanalysis/pyRPPG')
sys.path.append('/home/jagmohanmeher/nordlinglab-imageanalysis/pyRPPG/pyrppg')
sys.path.append('/home/jagmohanmeher/nordlinglab-imageanalysis/pyRPPG/tests')
import os, time, json
import cv2
import numpy as np
from scipy import signal, fft
import matplotlib.pyplot as plt
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from pyrppg.eval import findspikes, estimate_hr_snr_by_fft, estimate_hr_by_peaks, rolling_max, rolling_mean
from pyrppg.utils import FaceDetector, FaceAligner3D, set_exact_frame

assert len(sys.argv) == 2, 'Usage: python export_hr_gt.py list_xxx.json'

with open(sys.argv[1], 'r', encoding='utf-8') as f:
    exp = json.load(f)
    # for exp1 (list the test you want to process here)
    tests = ['static1', 'static2', 'static3', 'speak', 'rotate', 'bike1', 'bike2', 'bike3']
    cameras = ['camera1'] # preprocess camera1 only
    fs_ecg = 500
    root = exp['root'] if 'root' in exp else ''
    for subject in exp['subjects']:
        for test in tests:
            for camera in cameras:
                # check fields
                if 'id' not in subject: continue
                if test not in subject: continue
                if camera not in subject[test]: continue
                if 'video' not in subject[test][camera]: continue
                if 'bbox' not in subject[test][camera]: continue
                if 'colormap' not in subject[test][camera]: continue
                if 'aligned physio' not in subject[test][camera]: continue
                if 'gt' not in subject[test][camera]: continue
                if 'physio' not in subject[test]: continue
                if 'ecg avf' not in subject[test]['physio']: continue
                # TODO: check existence of files
                # start
                print(f'----- Subject {subject["id"]} in {test} with {camera} -----')
                tt = time.time()
                tes = subject[test]
                cam = subject[test][camera]
                # load aligned physio (requires the index of triggered frame (the first reading))
                try:
                    aligned_physio = np.loadtxt(os.path.join(root, cam['aligned physio']), delimiter=',')
                    istart, iend = int(aligned_physio[0, 0]), int(aligned_physio[-1, 0])+1
                    del aligned_physio
                except:
                    print('Failed to load aligned physio')
                    continue
                # load video
                cap = cv2.VideoCapture(os.path.join(root, cam['video']))
                fs = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                # load preprocessed bbox
                val = np.load(os.path.join(root, cam['bbox']))[:, 0]
                # get gtp raw (tpeaks)
                sos_ecg = signal.butter(6, [0.7, 3.5], 'bandpass', fs=fs_ecg, output='sos')
                ecg = np.loadtxt(os.path.join(root, tes['physio']['ecg avf']), delimiter=',', usecols=0)
                ipeaks = findspikes(ecg, 50, 20, 600).nonzero()[0]
                # plt.plot(ecg); plt.scatter(ipeaks, ecg[ipeaks], c='#FF0000'); plt.show()
                # all the t in the following use unit RelativeTime (1 RealtiveTime = 1/8000 second)
                t0 = np.loadtxt(os.path.join(root, tes['physio']['ecg avf']), delimiter=',', max_rows=1)[2]
                dt = 16 # 2ms = 16 * 1/8ms (500 Hz)
                tpeaks = t0 + ipeaks * dt
                t = np.loadtxt(os.path.join(root, cam['aligned physio']), delimiter=',', usecols=1) # time of frames from aligned physio
                # intervals
                L, dL = fft.next_fast_len(int(fs * 30)), int(fs * 5)
                hr_gt, hr_gtp = [], []
                valid = [] # by bbox (whether always has face in that interval)
                for i in range(0, iend - istart - L + 1, dL):
                    # by fft
                    x = rolling_max(ecg[i:i+L], wndlen=int(fs*0.2), center=True)
                    x = rolling_mean(x, wndlen=int(fs*0.2), center=True)
                    x = signal.sosfiltfilt(sos_ecg, x, padtype='odd')
                    # by fft (TODO: bug in this function? or just result is bad?)
                    hr_gt.append(estimate_hr_snr_by_fft(x, fs=fs_ecg, central_moment=True)[0])
                    # by peak
                    hr_gtp.append(estimate_hr_by_peaks(tpeaks, t[i], t[i+L-1], unit=8000.0))
                    valid.append(np.all(val[i:i+L] > 0)) # 1 for face detected, -1 for no
                    # print(f'hr (gt) = {hr_gt[-1] * 60:.3f}, hr (gtp) = {hr_gtp[-1] * 60:.3f}')
                hr_gt, hr_gtp = np.array(hr_gt, dtype=np.float64), np.array(hr_gtp, dtype=np.float64)
                np.savez(os.path.join(root, cam['gt']), hr_gt=hr_gt, hr_gtp=hr_gtp, valid=valid)
                print(f'Time cost: {time.time()-tt:.3f} seconds')
        