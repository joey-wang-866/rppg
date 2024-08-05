# get statistics (MAE, MAPE, ...) by HR from ground truth and HR and SNR from rPPG

import sys, os, json
import numpy as np
import matplotlib.pyplot as plt

tests = ['static1', 'static2', 'static3', 'speak', 'bike1', 'bike2', 'bike3']
camera = 'camera1'

with open(sys.argv[1], 'r') as f:
    exp = json.load(f)
    root = exp['root'] if 'root' in exp else ''
    for test in tests:
        ae_green, ae_ica, ae_chrom, ae_pos, ae_sph, ae_csc, ae_chromprn = [], [], [], [], [], [], []
        ape_green, ape_ica, ape_chrom, ape_pos, ape_sph, ape_csc, ape_chromprn = [], [], [], [], [], [], []
        snr_green, snr_ica, snr_chrom, snr_pos, snr_sph, snr_csc, snr_chromprn = [], [], [], [], [], [], []
        valid = []
        print(f'----- {test} -----')
        for subject in exp['subjects']:
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
            if 'hybrid' not in subject[test][camera]: continue
            print(f'-- Subject {subject["id"]} in {test} with {camera} --')
            cam = subject[test][camera]
            hybrid = np.load(os.path.join(root, cam['hybrid']))
            gt = np.load(os.path.join(root, cam['gt']))
            hr_green, hr_ica, hr_chrom, hr_pos, hr_sph, hr_csc, hr_chromprn = hybrid['hr_green'], hybrid['hr_ica'], hybrid['hr_chrom'], hybrid['hr_pos'], hybrid['hr_sph'], hybrid['hr_csc'], hybrid['hr_chromprn']
            sn_green, sn_ica, sn_chrom, sn_pos, sn_sph, sn_csc, sn_chromprn = hybrid['snr_green'], hybrid['snr_ica'], hybrid['snr_chrom'], hybrid['snr_pos'], hybrid['snr_sph'], hybrid['snr_csc'], hybrid['snr_chromprn']
            hr_gt, hr_gtp, vali = gt['hr_gt'], gt['hr_gtp'], gt['valid']
            n = min(len(hr_green), len(hr_ica), len(hr_chrom), len(hr_pos), len(hr_sph), len(hr_csc), len(hr_chromprn), len(hr_gtp))
            print(f'n = {n}')

            ae_green.append(np.abs(hr_green[:n] - hr_gtp[:n]))
            ae_ica.append(np.abs(hr_ica[:n] - hr_gtp[:n]))
            ae_chrom.append(np.abs(hr_chrom[:n] - hr_gtp[:n]))
            ae_pos.append(np.abs(hr_pos[:n] - hr_gtp[:n]))
            ae_sph.append(np.abs(hr_sph[:n] - hr_gtp[:n]))
            ae_csc.append(np.abs(hr_csc[:n] - hr_gtp[:n]))
            ae_chromprn.append(np.abs(hr_chromprn[:n] - hr_gtp[:n]))

            ape_green.append(np.abs(hr_green[:n] - hr_gtp[:n]) / hr_gtp[:n])
            ape_ica.append(np.abs(hr_ica[:n] - hr_gtp[:n]) / hr_gtp[:n])
            ape_chrom.append(np.abs(hr_chrom[:n] - hr_gtp[:n]) / hr_gtp[:n])
            ape_pos.append(np.abs(hr_pos[:n] - hr_gtp[:n]) / hr_gtp[:n])
            ape_sph.append(np.abs(hr_sph[:n] - hr_gtp[:n]) / hr_gtp[:n])
            ape_csc.append(np.abs(hr_csc[:n] - hr_gtp[:n]) / hr_gtp[:n])
            ape_chromprn.append(np.abs(hr_chromprn[:n] - hr_gtp[:n]) / hr_gtp[:n])

            snr_green.append(np.abs(sn_green[:n])) # 0.xxx should always > 0
            snr_ica.append(np.abs(sn_ica[:n]))
            snr_chrom.append(np.abs(sn_chrom[:n]))
            snr_pos.append(np.abs(sn_pos[:n]))
            snr_sph.append(np.abs(sn_sph[:n]))
            snr_csc.append(np.abs(sn_csc[:n]))
            snr_chromprn.append(np.abs(sn_chromprn[:n]))

            valid.append(vali)

        if len(ae_green) == 0: continue

        ae_green = np.concatenate(ae_green) * 60 # bpm
        ae_ica = np.concatenate(ae_ica) * 60
        ae_chrom = np.concatenate(ae_chrom) * 60
        ae_pos = np.concatenate(ae_pos) * 60
        ae_sph = np.concatenate(ae_sph) * 60
        ae_csc = np.concatenate(ae_csc) * 60
        ae_chromprn = np.concatenate(ae_chromprn) * 60

        ape_green = np.concatenate(ape_green) * 100 # 100%
        ape_ica = np.concatenate(ape_ica) * 100
        ape_chrom = np.concatenate(ape_chrom) * 100
        ape_pos = np.concatenate(ape_pos) * 100
        ape_sph = np.concatenate(ape_sph) * 100
        ape_csc = np.concatenate(ape_csc) * 100
        ape_chromprn = np.concatenate(ape_chromprn) * 100
        
        snr_green = np.log10(np.concatenate(snr_green)) * 10 # dB
        snr_ica = np.log10(np.concatenate(snr_ica)) * 10
        snr_chrom = np.log10(np.concatenate(snr_chrom)) * 10
        snr_pos = np.log10(np.concatenate(snr_pos)) * 10
        snr_sph = np.log10(np.concatenate(snr_sph)) * 10
        snr_csc = np.log10(np.concatenate(snr_csc)) * 10
        snr_chromprn = np.log10(np.concatenate(snr_chromprn)) * 10

        valid = np.concatenate(valid)

        # ae < 10 bpm, all
        print('For ae < 10 bpm (all)')
        mask_green = ae_green < 10
        mask_ica = ae_ica < 10
        mask_chrom = ae_chrom < 10
        mask_pos = ae_pos < 10
        mask_sph = ae_sph < 10
        mask_csc = ae_csc < 10
        mask_chromprn = ae_chromprn < 10

        sr_green = np.count_nonzero(mask_green) / len(mask_green) * 100 # success rate
        sr_ica = np.count_nonzero(mask_ica) / len(mask_ica) * 100 # success rate
        sr_chrom = np.count_nonzero(mask_chrom) / len(mask_chrom) * 100 # success rate
        sr_pos = np.count_nonzero(mask_pos) / len(mask_pos) * 100 # success rate
        sr_sph = np.count_nonzero(mask_sph) / len(mask_sph) * 100 # success rate
        sr_csc = np.count_nonzero(mask_csc) / len(mask_csc) * 100 # success rate
        sr_chromprn = np.count_nonzero(mask_chromprn) / len(mask_chromprn) * 100 # success rate

        mae_green = np.mean(ae_green[mask_green]) if sr_green > 0 else np.nan
        mae_ica = np.mean(ae_ica[mask_ica]) if sr_ica > 0 else np.nan
        mae_chrom = np.mean(ae_chrom[mask_chrom]) if sr_chrom > 0 else np.nan
        mae_pos = np.mean(ae_pos[mask_pos]) if sr_pos > 0 else np.nan
        mae_sph = np.mean(ae_sph[mask_sph]) if sr_sph > 0 else np.nan
        mae_csc = np.mean(ae_csc[mask_csc]) if sr_csc > 0 else np.nan
        mae_chromprn = np.mean(ae_chromprn[mask_chromprn]) if sr_chromprn > 0 else np.nan

        msnr_green = np.mean(snr_green[mask_green]) if sr_green > 0 else np.nan # geometric average
        msnr_ica = np.mean(snr_ica[mask_ica]) if sr_ica > 0 else np.nan # geometric average
        msnr_chrom = np.mean(snr_chrom[mask_chrom]) if sr_chrom > 0 else np.nan # geometric average
        msnr_pos = np.mean(snr_pos[mask_pos]) if sr_pos > 0 else np.nan # geometric average
        msnr_sph = np.mean(snr_sph[mask_sph]) if sr_sph > 0 else np.nan # geometric average
        msnr_csc = np.mean(snr_csc[mask_csc]) if sr_csc > 0 else np.nan # geometric average
        msnr_chromprn = np.mean(snr_chromprn[mask_chromprn]) if sr_chromprn > 0 else np.nan # geometric average

        print(f'     GREEN: MAE={mae_green:5.2f}bpm, SR={sr_green:.2f}%, SNR={msnr_green:.2f}dB')
        print(f'       ICA: MAE={mae_ica:5.2f}bpm, SR={sr_ica:.2f}%, SNR={msnr_ica:.2f}dB')
        print(f'     CHROM: MAE={mae_chrom:5.2f}bpm, SR={sr_chrom:.2f}%, SNR={msnr_chrom:.2f}dB')
        print(f'       POS: MAE={mae_pos:5.2f}bpm, SR={sr_pos:.2f}%, SNR={msnr_pos:.2f}dB')
        print(f'       SPH: MAE={mae_sph:5.2f}bpm, SR={sr_sph:.2f}%, SNR={msnr_sph:.2f}dB')
        print(f'       CSC: MAE={mae_csc:5.2f}bpm, SR={sr_csc:.2f}%, SNR={msnr_csc:.2f}dB')
        print(f'  CHROMPRN: MAE={mae_chromprn:5.2f}bpm, SR={sr_chromprn:.2f}%, SNR={msnr_chromprn:.2f}dB')
        plt.boxplot([ae_green[mask_green], ae_ica[mask_ica], ae_chrom[mask_chrom],
            ae_pos[mask_pos], ae_sph[mask_sph], ae_csc[mask_csc], ae_chromprn[mask_chromprn]],
            showfliers=False, labels=['GREEN', 'ICA', 'CHROM', 'POS', 'SPH', 'CSC', 'CHROM-PRN'])
        plt.title('MAE of HR')
        plt.ylabel('MAE (bpm)')
        plt.show()

        # # ae < 10 bpm, valid
        # print('For ae < 10 bpm (valid)')
        # mask_green = np.logical_and(ae_green < 10, valid)
        # mask_ica = np.logical_and(ae_ica < 10, valid)
        # mask_chrom = np.logical_and(ae_chrom < 10, valid)
        # mask_pos = np.logical_and(ae_pos < 10, valid)
        # mask_sph = np.logical_and(ae_sph < 10, valid)
        # mask_csc = np.logical_and(ae_csc < 10, valid)
        # mask_chromprn = np.logical_and(ae_chromprn < 10, valid)

        # sr_green = np.count_nonzero(mask_green) / np.count_nonzero(valid) * 100 # success arte
        # sr_ica = np.count_nonzero(mask_ica) / np.count_nonzero(valid) * 100 # success arte
        # sr_chrom = np.count_nonzero(mask_chrom) / np.count_nonzero(valid) * 100 # success arte
        # sr_pos = np.count_nonzero(mask_pos) / np.count_nonzero(valid) * 100 # success arte
        # sr_sph = np.count_nonzero(mask_sph) / np.count_nonzero(valid) * 100 # success arte
        # sr_csc = np.count_nonzero(mask_csc) / np.count_nonzero(valid) * 100 # success arte
        # sr_chromprn = np.count_nonzero(mask_chromprn) / np.count_nonzero(valid) * 100 # success arte

        # mae_green = np.mean(ae_green[mask_green]) if sr_green > 0 else np.nan
        # mae_ica = np.mean(ae_ica[mask_ica]) if sr_ica > 0 else np.nan
        # mae_chrom = np.mean(ae_chrom[mask_chrom]) if sr_chrom > 0 else np.nan
        # mae_pos = np.mean(ae_pos[mask_pos]) if sr_pos > 0 else np.nan
        # mae_sph = np.mean(ae_sph[mask_sph]) if sr_sph > 0 else np.nan
        # mae_csc = np.mean(ae_csc[mask_csc]) if sr_csc > 0 else np.nan
        # mae_chromprn = np.mean(ae_chromprn[mask_chromprn]) if sr_chromprn > 0 else np.nan

        # msnr_green = np.mean(snr_green[mask_green]) if sr_green > 0 else np.nan # geometric average
        # msnr_ica = np.mean(snr_ica[mask_ica]) if sr_ica > 0 else np.nan # geometric average
        # msnr_chrom = np.mean(snr_chrom[mask_chrom]) if sr_chrom > 0 else np.nan # geometric average
        # msnr_pos = np.mean(snr_pos[mask_pos]) if sr_pos > 0 else np.nan # geometric average
        # msnr_sph = np.mean(snr_sph[mask_sph]) if sr_sph > 0 else np.nan # geometric average
        # msnr_csc = np.mean(snr_csc[mask_csc]) if sr_csc > 0 else np.nan # geometric average
        # msnr_chromprn = np.mean(snr_chromprn[mask_chromprn]) if sr_chromprn > 0 else np.nan # geometric average

        # print(f'     GREEN: MAE={mae_green:5.2f}bpm, SR={sr_green:.2f}%, SNR={msnr_green:.2f}dB')
        # print(f'       ICA: MAE={mae_ica:5.2f}bpm, SR={sr_ica:.2f}%, SNR={msnr_ica:.2f}dB')
        # print(f'     CHROM: MAE={mae_chrom:5.2f}bpm, SR={sr_chrom:.2f}%, SNR={msnr_chrom:.2f}dB')
        # print(f'       POS: MAE={mae_pos:5.2f}bpm, SR={sr_pos:.2f}%, SNR={msnr_pos:.2f}dB')
        # print(f'       SPH: MAE={mae_sph:5.2f}bpm, SR={sr_sph:.2f}%, SNR={msnr_sph:.2f}dB')
        # print(f'       CSC: MAE={mae_csc:5.2f}bpm, SR={sr_csc:.2f}%, SNR={msnr_csc:.2f}dB')
        # print(f'  CHROMPRN: MAE={mae_chromprn:5.2f}bpm, SR={sr_chromprn:.2f}%, SNR={msnr_chromprn:.2f}dB')
        # plt.boxplot([ae_green[mask_green], ae_ica[mask_ica], ae_chrom[mask_chrom],
        #     ae_pos[mask_pos], ae_sph[mask_sph], ae_csc[mask_csc], ae_chromprn[mask_chromprn]],
        #     showfliers=False, labels=['GREEN', 'ICA', 'CHROM', 'POS', 'SPH', 'CSC', 'CHROM-PRN'])
        # plt.title('MAE of HR')
        # plt.ylabel('MAE (bpm)')
        # plt.show()

        # # ape < 10%, all
        # print('For ape < 10% (all)')
        # mask_green = ape_green < 10
        # mask_ica = ape_ica < 10
        # mask_chrom = ape_chrom < 10
        # mask_pos = ape_pos < 10
        # mask_sph = ape_sph < 10
        # mask_csc = ape_csc < 10
        # mask_chromprn = ape_chromprn < 10

        # sr_green = np.count_nonzero(mask_green) / len(mask_green) * 100 # success arte
        # sr_ica = np.count_nonzero(mask_ica) / len(mask_ica) * 100 # success arte
        # sr_chrom = np.count_nonzero(mask_chrom) / len(mask_chrom) * 100 # success arte
        # sr_pos = np.count_nonzero(mask_pos) / len(mask_pos) * 100 # success arte
        # sr_sph = np.count_nonzero(mask_sph) / len(mask_sph) * 100 # success arte
        # sr_csc = np.count_nonzero(mask_csc) / len(mask_csc) * 100 # success arte
        # sr_chromprn = np.count_nonzero(mask_chromprn) / len(mask_chromprn) * 100 # success arte

        # mape_green = np.mean(ape_green[mask_green]) if sr_green > 0 else np.nan
        # mape_ica = np.mean(ape_ica[mask_ica]) if sr_ica > 0 else np.nan
        # mape_chrom = np.mean(ape_chrom[mask_chrom]) if sr_chrom > 0 else np.nan
        # mape_pos = np.mean(ape_pos[mask_pos]) if sr_pos > 0 else np.nan
        # mape_sph = np.mean(ape_sph[mask_sph]) if sr_sph > 0 else np.nan
        # mape_csc = np.mean(ape_csc[mask_csc]) if sr_csc > 0 else np.nan
        # mape_chromprn = np.mean(ape_chromprn[mask_chromprn]) if sr_chromprn > 0 else np.nan

        # msnr_green = np.mean(snr_green[mask_green]) if sr_green > 0 else np.nan # geometric average
        # msnr_ica = np.mean(snr_ica[mask_ica]) if sr_ica > 0 else np.nan # geometric average
        # msnr_chrom = np.mean(snr_chrom[mask_chrom]) if sr_chrom > 0 else np.nan # geometric average
        # msnr_pos = np.mean(snr_pos[mask_pos]) if sr_pos > 0 else np.nan # geometric average
        # msnr_sph = np.mean(snr_sph[mask_sph]) if sr_sph > 0 else np.nan # geometric average
        # msnr_csc = np.mean(snr_csc[mask_csc]) if sr_csc > 0 else np.nan # geometric average
        # msnr_chromprn = np.mean(snr_chromprn[mask_chromprn]) if sr_chromprn > 0 else np.nan # geometric average

        # print(f'     GREEN: MAPE={mape_green:5.2f}%, SR={sr_green:.2f}%, SNR={msnr_green:.2f}dB')
        # print(f'       ICA: MAPE={mape_ica:5.2f}%, SR={sr_ica:.2f}%, SNR={msnr_ica:.2f}dB')
        # print(f'     CHROM: MAPE={mape_chrom:5.2f}%, SR={sr_chrom:.2f}%, SNR={msnr_chrom:.2f}dB')
        # print(f'       POS: MAPE={mape_pos:5.2f}%, SR={sr_pos:.2f}%, SNR={msnr_pos:.2f}dB')
        # print(f'       SPH: MAPE={mape_sph:5.2f}%, SR={sr_sph:.2f}%, SNR={msnr_sph:.2f}dB')
        # print(f'       CSC: MAPE={mape_csc:5.2f}%, SR={sr_csc:.2f}%, SNR={msnr_csc:.2f}dB')
        # print(f'  CHROMPRN: MAPE={mape_chromprn:5.2f}%, SR={sr_chromprn:.2f}%, SNR={msnr_chromprn:.2f}dB')
        # plt.boxplot([ape_green[mask_green], ape_ica[mask_ica], ape_chrom[mask_chrom],
        #     ape_pos[mask_pos], ape_sph[mask_sph], ape_csc[mask_csc], ape_chromprn[mask_chromprn]],
        #     showfliers=False, labels=['GREEN', 'ICA', 'CHROM', 'POS', 'SPH', 'CSC', 'CHROM-PRN'])
        # plt.title('MAPE of HR')
        # plt.ylabel('MAPE (%)')
        # plt.show()

        # # ape < 10%, valid
        # print('For ape < 10% (valid)')
        # mask_green = np.logical_and(ape_green < 10, valid)
        # mask_ica = np.logical_and(ape_ica < 10, valid)
        # mask_chrom = np.logical_and(ape_chrom < 10, valid)
        # mask_pos = np.logical_and(ape_pos < 10, valid)
        # mask_sph = np.logical_and(ape_sph < 10, valid)
        # mask_csc = np.logical_and(ape_csc < 10, valid)
        # mask_chromprn = np.logical_and(ape_chromprn < 10, valid)

        # sr_green = np.count_nonzero(mask_green) / np.count_nonzero(valid) * 100 # success arte
        # sr_ica = np.count_nonzero(mask_ica) / np.count_nonzero(valid) * 100 # success arte
        # sr_chrom = np.count_nonzero(mask_chrom) / np.count_nonzero(valid) * 100 # success arte
        # sr_pos = np.count_nonzero(mask_pos) / np.count_nonzero(valid) * 100 # success arte
        # sr_sph = np.count_nonzero(mask_sph) / np.count_nonzero(valid) * 100 # success arte
        # sr_csc = np.count_nonzero(mask_csc) / np.count_nonzero(valid) * 100 # success arte
        # sr_chromprn = np.count_nonzero(mask_chromprn) / np.count_nonzero(valid) * 100 # success arte

        # mape_green = np.mean(ape_green[mask_green]) if sr_green > 0 else np.nan
        # mape_ica = np.mean(ape_ica[mask_ica]) if sr_ica > 0 else np.nan
        # mape_chrom = np.mean(ape_chrom[mask_chrom]) if sr_chrom > 0 else np.nan
        # mape_pos = np.mean(ape_pos[mask_pos]) if sr_pos > 0 else np.nan
        # mape_sph = np.mean(ape_sph[mask_sph]) if sr_sph > 0 else np.nan
        # mape_csc = np.mean(ape_csc[mask_csc]) if sr_csc > 0 else np.nan
        # mape_chromprn = np.mean(ape_chromprn[mask_chromprn]) if sr_chromprn > 0 else np.nan

        # msnr_green = np.mean(snr_green[mask_green]) if sr_green > 0 else np.nan # geometric average
        # msnr_ica = np.mean(snr_ica[mask_ica]) if sr_ica > 0 else np.nan # geometric average
        # msnr_chrom = np.mean(snr_chrom[mask_chrom]) if sr_chrom > 0 else np.nan # geometric average
        # msnr_pos = np.mean(snr_pos[mask_pos]) if sr_pos > 0 else np.nan # geometric average
        # msnr_sph = np.mean(snr_sph[mask_sph]) if sr_sph > 0 else np.nan # geometric average
        # msnr_csc = np.mean(snr_csc[mask_csc]) if sr_csc > 0 else np.nan # geometric average
        # msnr_chromprn = np.mean(snr_chromprn[mask_chromprn]) if sr_chromprn > 0 else np.nan # geometric average

        # print(f'     GREEN: MAPE={mape_green:5.2f}%, SR={sr_green:.2f}%, SNR={msnr_green:.2f}dB')
        # print(f'       ICA: MAPE={mape_ica:5.2f}%, SR={sr_ica:.2f}%, SNR={msnr_ica:.2f}dB')
        # print(f'     CHROM: MAPE={mape_chrom:5.2f}%, SR={sr_chrom:.2f}%, SNR={msnr_chrom:.2f}dB')
        # print(f'       POS: MAPE={mape_pos:5.2f}%, SR={sr_pos:.2f}%, SNR={msnr_pos:.2f}dB')
        # print(f'       SPH: MAPE={mape_sph:5.2f}%, SR={sr_sph:.2f}%, SNR={msnr_sph:.2f}dB')
        # print(f'       CSC: MAPE={mape_csc:5.2f}%, SR={sr_csc:.2f}%, SNR={msnr_csc:.2f}dB')
        # print(f'  CHROMPRN: MAPE={mape_chromprn:5.2f}%, SR={sr_chromprn:.2f}%, SNR={msnr_chromprn:.2f}dB')
        # plt.boxplot([ape_green[mask_green], ape_ica[mask_ica], ape_chrom[mask_chrom],
        #     ape_pos[mask_pos], ape_sph[mask_sph], ape_csc[mask_csc], ape_chromprn[mask_chromprn]],
        #     showfliers=False, labels=['GREEN', 'ICA', 'CHROM', 'POS', 'SPH', 'CSC', 'CHROM-PRN'])
        # plt.title('MAPE of HR')
        # plt.ylabel('MAPE (%)')
        # plt.show()