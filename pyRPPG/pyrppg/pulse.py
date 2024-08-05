from .utils import *
from .eval import *
from scipy import signal


# class HBCGPCA:
#     ''' Motion-based mtehod
#     Balakrishnan, G., Durand, F., & Guttag, J. (2013). Detecting pulse from head motions in video. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 3430-3437).
#     '''


# class HBCGICA:
#     ''' Motion-based mtehod
#     Shan, L., & Yu, M. (2013, December). Video-based heart rate measurement using head motion tracking and ICA. In 2013 6th International Congress on Image and Signal Processing (CISP) (Vol. 1, pp. 160-164). IEEE.
#     '''


class RPPGGREEN:
    """
    A color-based method for measuring blood volume pulse from the green channel of facial video.

    Details of our implementation...

    Reference:
        Verkruysse, W., Svaasand, L. O., & Nelson, J. S. (2008). Remote plethysmographic imaging using ambient light. Optics express, 16(26), 21434-21445.
    """

    def __init__(self, face_detector=None):
        """
        Constructor.

        :param face_detector: FaceDetector instance (default: None).
        :type face_detector: FaceDetector, optional
        """
        self.face_detector = face_detector
        # self.skin_detector = SkinDetector() if skin_detector is None else skin_detector
        self.roi = np.array([  # xmin, ymin, xmax, ymax (x, y => [0, 1])
            [0.35, 0.22, 0.65, 0.30],  # forehead (good)
            [0.49, 0.27, 0.51, 0.28],  # forehead point (not good)
            [0.80, 0.20, 0.95, 0.45],  # left temple (not good)
            [0.05, 0.05, 0.95, 0.95],  # full face (better?)
            # [0.66, 0.45, 0.90, 0.60], # left cheek (not good)
            [0.66, 0.48, 0.90, 0.65],
            # [0.10, 0.45, 0.34, 0.60] # right cheek (not good)
            [0.10, 0.48, 0.34, 0.65],
            # mouth (best? if static, currently this is more stable)
            [0.30, 0.70, 0.70, 0.87]
        ], dtype=np.float32)  # by left and right cheek is best?

    def extract(self, cap, start_frame=0, end_frame=None, offset=0, rotate=None, resize_shape=None, is_dark=False, imputation=False, display=False, preprocessed_bboxes=None):
        """
        Extracts the signal from the video.

        :param cap: VideoCapture object for the input video
        :type cap: cv2.VideoCapture
        :param start_frame: Starting frame index (default: 0).
        :type start_frame: int, optional
        :param end_frame: Ending frame index (default: None).
        :type end_frame: int, optional
        :param offset: Frame offset (default: 0).
        :type offset: int, optional
        :param rotate: Rotation flag for the input image (default: None).
        :type rotate: cv2.RotateFlags, optional
        :param resize_shape: Resize shape for the input image (default: None).
        :type resize_shape: tuple, optional
        :param is_dark: Whether the input image is dark (default: False).
        :type is_dark: bool, optional
        :param imputation: Whether to perform imputation for missing values (default: False).
        :type imputation: bool, optional
        :param display: Whether to display the image with ROI rectangles (default: False).
        :type display: bool, optional
        :param preprocessed_bboxes: Preprocessed bounding boxes (default: None).
        :type preprocessed_bboxes: list, optional
        :return: Processed signal
        :rtype: numpy.ndarray
        """
        if end_frame is None:
            # might be inaccurate
            end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if preprocessed_bboxes is None and self.face_detector is None:
            self.face_detector = FaceDetector()
        icurr = cap.get(cv2.CAP_PROP_POS_FRAMES)
        s = np.full(end_frame - start_frame, np.nan, dtype=np.float64)
        set_exact_frame(cap, start_frame + offset)
        for i in range(end_frame - start_frame):
            ret, img = cap.read()
            if not ret:
                break
            if rotate is not None:
                img = cv2.rotate(img, rotate)
            if resize_shape is not None:
                img = cv2.resize(img, resize_shape)
            bboxes = self.face_detector.detect(
                img) if preprocessed_bboxes is None else preprocessed_bboxes[start_frame + i]
            if bboxes is None:
                print('No face is detected in frame %d' % (start_frame + i))
                continue
            x1, y1, x2, y2 = tuple(bboxes[0])
            w, h = x2 - x1, y2 - y1
            rois = np.rint(np.array(
                [x1, y1, x1, y1]) + np.array([w, h, w, h]) * self.roi).astype(np.int32)
            imgw, imgh = img.shape[:2]
            np.clip(rois, np.zeros(4, dtype=np.int32), np.array(
                [imgw, imgh, imgw, imgh]), out=rois)  # inplace
            s[i] = 0
            s[i] += np.mean(img[rois[0, 1]:rois[0, 3],
                            rois[0, 0]:rois[0, 2], 1])
            s[i] += np.mean(img[rois[6, 1]:rois[6, 3],
                            rois[6, 0]:rois[6, 2], 1])
            if display:
                tmp = img.copy()
                cv2.rectangle(tmp, (x1, y1), (x2, y2), (0, 255, 0), 1)
                for roi in rois:
                    cv2.rectangle(tmp, tuple(roi[:2]), tuple(
                        roi[2:]), (0, 0, 255), 1)
                cv2.imshow('rois', tmp)
                if cv2.waitKey(1) != -1:
                    break
        if display:
            cv2.destroyWindow('rois')
        if np.any(np.isnan(s)):
            if imputation:
                valid = np.isfinite(s)
                index = np.arange(len(s))
                s = interp1d(index[valid], s[valid], kind='cubic',
                             bounds_error=False, fill_value='extrapolate')(index)
            else:  # has missing value, but didn't deal with it
                print('NaN in signal.')
                return None
        sos = signal.butter(6, [0.7, 3.5], 'bandpass',
                            fs=cap.get(cv2.CAP_PROP_FPS), output='sos')
        sf = signal.sosfiltfilt(sos, s, padtype='odd')
        set_exact_frame(cap, icurr)
        return sf


class RPPGPCA:
    """
    Color-based method for measuring blood volume pulse using Principal Component Analysis (PCA).

    Reference:
        Lewandowska, M., Rumiński, J., Kocejko, T., & Nowak, J. (2011, September). Measuring pulse rate with a webcam—a non-contact method for evaluating cardiac activity. In 2011 federated conference on computer science and information systems (FedCSIS) (pp. 405-410). IEEE.
    """

    def __init__(self, face_detector=None, skin_detector=None):
        """
        Constructor.

        :param face_detector: FaceDetector instance (default: None).
        :type face_detector: FaceDetector, optional
        :param skin_detector: SkinDetector instance (default: None).
        :type skin_detector: SkinDetector, optional
        """
        self.face_detector = face_detector
        self.skin_detector = SkinDetector() if skin_detector is None else skin_detector

    def extract(self, cap, start_frame=0, end_frame=None, offset=0, rotate=None, resize=None, is_dark=False, imputation=False, display=False, preprocessed_bboxes=None):
        """
        Extracts the signal from the video.

        :param cap: VideoCapture object for the input video
        :type cap: cv2.VideoCapture
        :param start_frame: Starting frame index (default: 0).
        :type start_frame: int, optional
        :param end_frame: Ending frame index (default: None).
        :type end_frame: int, optional
        :param offset: Frame offset (default: 0).
        :type offset: int, optional
        :param rotate: Rotation flag for the input image (default: None).
        :type rotate: cv2.RotateFlags, optional
        :param resize: Resize shape for the input image (default: None).
        :type resize: tuple, optional
        :param is_dark: Whether the input image is dark (default: False).
        :type is_dark: bool, optional
        :param imputation: Whether to perform imputation for missing values (default: False).
        :type imputation: bool, optional
        :param display: Whether to display the image with ROI rectangles (default: False).
        :type display: bool, optional
        :param preprocessed_bboxes: Preprocessed bounding boxes (default: None).
        :type preprocessed_bboxes: list, optional
        :return: Processed signal
        :rtype: numpy.ndarray
        """
        if end_frame is None:
            # might be inaccurate
            end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if preprocessed_bboxes is None and self.face_detector is None:
            self.face_detector = FaceDetector()
        icurr = cap.get(cv2.CAP_PROP_POS_FRAMES)
        bgr = np.full((end_frame - start_frame, 3), np.nan, dtype=np.float64)
        set_exact_frame(cap, start_frame + offset)
        for i in range(end_frame - start_frame):
            ret, img = cap.read()
            if not ret:
                break
            if rotate is not None:
                img = cv2.rotate(img, rotate)
            if resize is not None:
                img = cv2.resize(img, resize)
            bboxes = self.face_detector.detect(
                img) if preprocessed_bboxes is None else preprocessed_bboxes[start_frame + i]
            if bboxes is None:
                print('No face is detected in frame %d' % (start_frame + i))
                continue
            x1, y1, x2, y2 = tuple(bboxes[0])
            y1, y2 = (5 * y1 + 5 * y2) // 10, (3 *
                                               y1 + 7 * y2) // 10  # cheeks region
            face = img[y1:y2, x1:x2]
            mask = self.skin_detector.detect(face, is_dark=is_dark)
            bgr[i] = cv2.mean(face, mask=mask)[:3]
            if display:
                tmp = cv2.bitwise_and(face, face, mask=mask)
                cv2.imshow('skin', tmp)
                if cv2.waitKey(1) != -1:
                    break
        if display:
            cv2.destroyWindow('skin')
        if np.any(np.isnan(bgr)):
            if imputation:
                valid = np.isfinite(bgr)
                index = np.arange(len(bgr))
                bgr = interp1d(index[valid], bgr[valid], kind='cubic',
                               bounds_error=False, fill_value='extrapolate')(index)
            else:  # has missing value, but didn't deal with it
                print('NaN in signal.')
                return None
        sos = signal.butter(6, [0.7, 3.5], 'bandpass',
                            fs=cap.get(cv2.CAP_PROP_FPS), output='sos')
        bgrf = signal.sosfiltfilt(sos, bgr, padtype='odd', axis=0)
        mean, eigvec, eigval = cv2.PCACompute2(
            bgrf, mean=None, maxComponents=3)
        s = np.matmul(bgrf, eigvec.T)[:, 0]  # the first component
        sf = signal.sosfiltfilt(sos, s, padtype='odd')
        set_exact_frame(cap, icurr)
        return sf


class RPPGICA:
    ''' 
    Color-based method for measuring blood volume pulse using Independent Component Analysis (ICA).

    Reference:
        Poh, M. Z., McDuff, D. J., & Picard, R. W. (2010). Non-contact, automated cardiac pulse measurements using video imaging and blind source separation. Optics express, 18(10), 10762-10774.
    '''

    def __init__(self, face_detector=None, skin_detector=None):
        """
        Constructor.

        :param face_detector: FaceDetector instance (default: None).
        :type face_detector: FaceDetector, optional
        :param skin_detector: SkinDetector instance (default: None).
        :type skin_detector: SkinDetector, optional
        """
        self.face_detector = face_detector
        self.skin_detector = SkinDetector() if skin_detector is None else skin_detector
        self.transformer = ICA(n_components=3, backend=ICA.BACKEND_JADE)

    def extract(self, cap, start_frame=0, end_frame=None, offset=0, rotate=None, resize=None, is_dark=False, imputation=False, display=False, preprocessed_bboxes=None):
        """
        Extracts the signal from the video.

        :param cap: VideoCapture object for the input video
        :type cap: cv2.VideoCapture
        :param start_frame: Starting frame index (default: 0).
        :type start_frame: int, optional
        :param end_frame: Ending frame index (default: None).
        :type end_frame: int, optional
        :param offset: Frame offset (default: 0).
        :type offset: int, optional
        :param rotate: Rotation flag for the input image (default: None).
        :type rotate: cv2.RotateFlags, optional
        :param resize: Resize shape for the input image (default: None).
        :type resize: tuple, optional
        :param is_dark: Whether the input image is dark (default: False).
        :type is_dark: bool, optional
        :param imputation: Whether to perform imputation for missing values (default: False).
        :type imputation: bool, optional
        :param display: Whether to display the image with ROI rectangles (default: False).
        :type display: bool, optional
        :param preprocessed_bboxes: Preprocessed bounding boxes (default: None).
        :type preprocessed_bboxes: list, optional
        :return: Processed signal
        :rtype: numpy.ndarray
        """
        if end_frame is None:
            # might be inaccurate
            end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if preprocessed_bboxes is None and self.face_detector is None:
            self.face_detector = FaceDetector()
        icurr = cap.get(cv2.CAP_PROP_POS_FRAMES)
        bgr = np.full((end_frame - start_frame, 3), np.nan, dtype=np.float64)
        set_exact_frame(cap, start_frame + offset)
        for i in range(end_frame - start_frame):
            ret, img = cap.read()
            if not ret:
                break
            if rotate is not None:
                img = cv2.rotate(img, rotate)
            if resize is not None:
                img = cv2.resize(img, resize)
            bboxes = self.face_detector.detect(
                img) if preprocessed_bboxes is None else preprocessed_bboxes[start_frame + i]
            if bboxes is None:
                print('No face is detected in frame %d' % (start_frame + i))
                continue
            x1, y1, x2, y2 = tuple(bboxes[0])
            y1, y2 = (5 * y1 + 5 * y2) // 10, (3 *
                                               y1 + 7 * y2) // 10  # cheeks region
            face = img[y1:y2, x1:x2]
            mask = self.skin_detector.detect(face, is_dark=is_dark)
            bgr[i] = cv2.mean(face, mask=mask)[:3]
            if display:
                tmp = cv2.bitwise_and(face, face, mask=mask)
                cv2.imshow('skin', tmp)
                if cv2.waitKey(1) != -1:
                    break
        if display:
            cv2.destroyWindow('skin')
        if np.any(np.isnan(bgr)):
            if imputation:
                valid = np.isfinite(bgr)
                index = np.arange(len(bgr))
                bgr = interp1d(index[valid], bgr[valid], kind='cubic',
                               bounds_error=False, fill_value='extrapolate')(index)
            else:  # has missing value, but didn't deal with it
                print('NaN in signal.')
                return None
        sos = signal.butter(6, [0.7, 3.5], 'bandpass',
                            fs=cap.get(cv2.CAP_PROP_FPS), output='sos')
        bgrf = signal.sosfiltfilt(sos, bgr, padtype='odd', axis=0)
        # zero mean, unit variance (already in JADE), and JADE sort in the end
        # Poh2010 said they usually select the second component
        s = self.transformer.fit_transform(bgrf)[:, 1]
        sf = signal.sosfiltfilt(sos, s, padtype='odd')
        set_exact_frame(cap, icurr)
        return sf


class RPPGCHROM:
    ''' 
    Color-based method for measuring blood volume pulse using Chrominance model.

    Reference:
        De Haan, G., & Jeanne, V. (2013). Robust pulse rate from chrominance-based rPPG. IEEE Transactions on Biomedical Engineering, 60(10), 2878-2886.
    '''

    def __init__(self, face_detector=None, skin_detector=None):
        """
        Constructor.

        :param face_detector: FaceDetector instance (default: None).
        :type face_detector: FaceDetector, optional
        :param skin_detector: SkinDetector instance (default: None).
        :type skin_detector: SkinDetector, optional
        """
        self.face_detector = face_detector
        self.skin_detector = SkinDetector() if skin_detector is None else skin_detector

    def extract(self, cap, start_frame=0, end_frame=None, offset=0, rotate=None, resize=None, is_dark=False, imputation=False, display=False, preprocessed_bboxes=None):
        """
        Extracts the signal from the video.

        :param cap: VideoCapture object for the input video
        :type cap: cv2.VideoCapture
        :param start_frame: Starting frame index (default: 0).
        :type start_frame: int, optional
        :param end_frame: Ending frame index (default: None).
        :type end_frame: int, optional
        :param offset: Frame offset (default: 0).
        :type offset: int, optional
        :param rotate: Rotation flag for the input image (default: None).
        :type rotate: cv2.RotateFlags, optional
        :param resize: Resize shape for the input image (default: None).
        :type resize: tuple, optional
        :param is_dark: Whether the input image is dark (default: False).
        :type is_dark: bool, optional
        :param imputation: Whether to perform imputation for missing values (default: False).
        :type imputation: bool, optional
        :param display: Whether to display the image with ROI rectangles (default: False).
        :type display: bool, optional
        :param preprocessed_bboxes: Preprocessed bounding boxes (default: None).
        :type preprocessed_bboxes: list, optional
        :return: Processed signal
        :rtype: numpy.ndarray
        """
        if end_frame is None:
            # might be inaccurate
            end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if preprocessed_bboxes is None and self.face_detector is None:
            self.face_detector = FaceDetector()
        icurr = cap.get(cv2.CAP_PROP_POS_FRAMES)
        fs = cap.get(cv2.CAP_PROP_FPS)
        bgr = np.full((end_frame - start_frame, 3), np.nan, dtype=np.float64)
        set_exact_frame(cap, start_frame + offset)
        for i in range(end_frame - start_frame):
            ret, img = cap.read()
            if not ret:
                break
            if rotate is not None:
                img = cv2.rotate(img, rotate)
            if resize is not None:
                img = cv2.resize(img, resize)
            bboxes = self.face_detector.detect(
                img) if preprocessed_bboxes is None else preprocessed_bboxes[start_frame + i]
            if bboxes is None:
                print('No face is detected in frame %d' % (start_frame + i))
                continue
            x1, y1, x2, y2 = tuple(bboxes[0])
            face = img[y1:y2, x1:x2]
            mask = self.skin_detector.detect(face, is_dark=is_dark)
            bgr[i] = cv2.mean(face, mask=mask)[:3]
            if display:
                tmp = cv2.bitwise_and(face, face, mask=mask)
                cv2.imshow('skin', tmp)
                if cv2.waitKey(1) != -1:
                    break
        if display:
            cv2.destroyWindow('skin')
        if np.any(np.isnan(bgr)):
            if imputation:
                valid = np.isfinite(bgr)
                index = np.arange(len(bgr))
                bgr = interp1d(index[valid], bgr[valid], kind='cubic',
                               bounds_error=False, fill_value='extrapolate')(index)
            else:  # has missing value, but didn't deal with it
                print('NaN in signal.')
                return None
        interval = int(fs * 32 / 20)
        sos = signal.butter(6, [0.7, 3.5], 'bandpass', fs=fs, output='sos')
        wnd = signal.windows.hamming(interval)
        s = np.zeros(end_frame - start_frame, dtype=np.float64)
        bgrn = bgr / \
            rolling_mean(bgr, window_len=int(fs*2), axis=0, center=True)
        # i+interval <= end_frame - start_frame (i < end_frame - start_frame - interval + 1)
        for i in range(0, end_frame - start_frame - interval + 1, interval // 2):
            bn, gn, rn = bgrn[i:i+interval, 0], bgrn[i:i +
                                                     interval, 1], bgrn[i:i+interval, 2]
            xs = 3 * rn - 2 * gn  # 3Rn - 2Gn
            ys = 1.5 * rn + gn - 1.5 * bn  # 1.5Rn + Gn - 1.5Bn
            xf = signal.sosfiltfilt(sos, xs, padtype='odd')
            yf = signal.sosfiltfilt(sos, ys, padtype='odd')
            alpha = np.std(xf) / np.std(yf)
            s[i:i+interval] += (xf - alpha * yf) * wnd
        sf = signal.sosfiltfilt(sos, s, padtype='odd')
        set_exact_frame(cap, icurr)
        return sf


class RPPGCHROMICA:
    ''' 
    Color-based method for measuring blood volume pulse using Chrominance model and ICA with overlap-adding.

    Reference:
        De Haan, G., & Jeanne, V. (2013). Robust pulse rate from chrominance-based rPPG. IEEE Transactions on Biomedical Engineering, 60(10), 2878-2886.
    '''

    def __init__(self, face_detector=None, skin_detector=None):
        """
        Constructor.

        :param face_detector: FaceDetector instance (default: None).
        :type face_detector: FaceDetector, optional
        :param skin_detector: SkinDetector instance (default: None).
        :type skin_detector: SkinDetector, optional
        """
        self.face_detector = face_detector
        self.skin_detector = SkinDetector() if skin_detector is None else skin_detector
        self.transformer = ICA(n_components=3, backend=ICA.BACKEND_JADE)

    def extract(self, cap, start_frame=0, end_frame=None, offset=0, rotate=None, resize=None, is_dark=False, imputation=False, display=False, preprocessed_bboxes=None):
        """
        Extracts the signal from the video.

        :param cap: VideoCapture object for the input video
        :type cap: cv2.VideoCapture
        :param start_frame: Starting frame index (default: 0).
        :type start_frame: int, optional
        :param end_frame: Ending frame index (default: None).
        :type end_frame: int, optional
        :param offset: Frame offset (default: 0).
        :type offset: int, optional
        :param rotate: Rotation flag for the input image (default: None).
        :type rotate: cv2.RotateFlags, optional
        :param resize: Resize shape for the input image (default: None).
        :type resize: tuple, optional
        :param is_dark: Whether the input image is dark (default: False).
        :type is_dark: bool, optional
        :param imputation: Whether to perform imputation for missing values (default: False).
        :type imputation: bool, optional
        :param display: Whether to display the image with ROI rectangles (default: False).
        :type display: bool, optional
        :param preprocessed_bboxes: Preprocessed bounding boxes (default: None).
        :type preprocessed_bboxes: list, optional
        :return: Processed signal
        :rtype: numpy.ndarray
        """
        if end_frame is None:
            # might be inaccurate
            end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if preprocessed_bboxes is None and self.face_detector is None:
            self.face_detector = FaceDetector()
        icurr = cap.get(cv2.CAP_PROP_POS_FRAMES)
        fs = cap.get(cv2.CAP_PROP_FPS)
        bgr = np.full((end_frame - start_frame, 3), np.nan, dtype=np.float64)
        set_exact_frame(cap, start_frame + offset)
        for i in range(end_frame - start_frame):
            ret, img = cap.read()
            if not ret:
                break
            if rotate is not None:
                img = cv2.rotate(img, rotate)
            if resize is not None:
                img = cv2.resize(img, resize)
            bboxes = self.face_detector.detect(
                img) if preprocessed_bboxes is None else preprocessed_bboxes[start_frame + i]
            if bboxes is None:
                print('No face is detected in frame %d' % (start_frame + i))
                continue
            x1, y1, x2, y2 = tuple(bboxes[0])
            face = img[y1:y2, x1:x2]
            mask = self.skin_detector.detect(face, is_dark=is_dark)
            bgr[i] = cv2.mean(face, mask=mask)[:3]
            if display:
                tmp = cv2.bitwise_and(face, face, mask=mask)
                cv2.imshow('skin', tmp)
                if cv2.waitKey(1) != -1:
                    break
        if display:
            cv2.destroyWindow('skin')
        if np.any(np.isnan(bgr)):
            if imputation:
                valid = np.isfinite(bgr)
                index = np.arange(len(bgr))
                bgr = interp1d(index[valid], bgr[valid], kind='cubic',
                               bounds_error=False, fill_value='extrapolate')(index)
            else:  # has missing value, but didn't deal with it
                print('NaN in signal.')
                return None
        interval = int(fs * 128 / 20)
        sos = signal.butter(6, [0.7, 3.5], 'bandpass', fs=fs, output='sos')
        wnd = signal.windows.hamming(interval)
        s = np.zeros(end_frame - start_frame, dtype=np.float64)
        bgrn_all = bgr / \
            rolling_mean(bgr, window_len=int(fs*2), axis=0, center=True)
        for i in range(0, end_frame - start_frame - interval + 1, interval // 2):
            x = self.transformer.fit_transform(bgrn_all[i:i+interval])
            xf = signal.sosfiltfilt(sos, x, padtype='odd', axis=0)
            snr = estimate_hr_snr_by_fft(xf, fs, axis=0)[
                1]  # different here, but similar
            s[i:i+interval] += xf[:, np.argmax(snr)] * wnd
        sf = signal.sosfiltfilt(sos, s, padtype='odd')
        set_exact_frame(cap, icurr)
        return sf


# class RPPGPBV:
#     ''' Color-based method
#     De Haan, G., & Van Leest, A. (2014). Improved motion robustness of remote-PPG by using the blood volume pulse signature. Physiological measurement, 35(9), 1913.
#     '''


# class RPPGSAMC:
#     ''' Color-based method
#     Tulyakov, S., Alameda-Pineda, X., Ricci, E., Yin, L., Cohn, J. F., & Sebe, N. (2016). Self-adaptive matrix completion for heart rate estimation from face videos under realistic conditions. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2396-2404).
#     '''


class RPPGSSR:
    ''' 
    Color-based method for measuring blood volume pulse using spatial subspace rotation (SSR).

    Reference:
        Wang, W., Stuijk, S., & De Haan, G. (2015). A novel algorithm for remote photoplethysmography: Spatial subspace rotation. IEEE transactions on biomedical engineering, 63(9), 1974-1984.
    '''

    def __init__(self, face_detector=None, skin_detector=None):
        """
        Constructor.

        :param face_detector: FaceDetector instance (default: None).
        :type face_detector: FaceDetector, optional
        :param skin_detector: SkinDetector instance (default: None).
        :type skin_detector: SkinDetector, optional
        """
        self.face_detector = face_detector
        self.skin_detector = SkinDetector() if skin_detector is None else skin_detector

    def extract(self, cap, start_frame=0, end_frame=None, offset=0, rotate=None, resize=None, is_dark=False, imputation=False, display=False, preprocessed_bboxes=None):
        """
        Extracts the signal from the video.

        :param cap: VideoCapture object for the input video
        :type cap: cv2.VideoCapture
        :param start_frame: Starting frame index (default: 0).
        :type start_frame: int, optional
        :param end_frame: Ending frame index (default: None).
        :type end_frame: int, optional
        :param offset: Frame offset (default: 0).
        :type offset: int, optional
        :param rotate: Rotation flag for the input image (default: None).
        :type rotate: cv2.RotateFlags, optional
        :param resize: Resize shape for the input image (default: None).
        :type resize: tuple, optional
        :param is_dark: Whether the input image is dark (default: False).
        :type is_dark: bool, optional
        :param imputation: Whether to perform imputation for missing values (default: False).
        :type imputation: bool, optional
        :param display: Whether to display the image with ROI rectangles (default: False).
        :type display: bool, optional
        :param preprocessed_bboxes: Preprocessed bounding boxes (default: None).
        :type preprocessed_bboxes: list, optional
        :return: Processed signal
        :rtype: numpy.ndarray
        """
        if end_frame is None:
            # might be inaccurate
            end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if preprocessed_bboxes is None and self.face_detector is None:
            self.face_detector = FaceDetector()
        icurr = cap.get(cv2.CAP_PROP_POS_FRAMES)
        fs = cap.get(cv2.CAP_PROP_FPS)
        # eigvals for every frame
        D = np.full((end_frame - start_frame, 3), np.nan, dtype=np.float64)
        U = np.full((end_frame - start_frame, 3, 3), np.nan,
                    dtype=np.float64)  # eigvecs for every frame
        set_exact_frame(cap, start_frame + offset)
        for i in range(end_frame - start_frame):
            ret, img = cap.read()
            if not ret:
                break
            if rotate is not None:
                img = cv2.rotate(img, rotate)
            if resize is not None:
                img = cv2.resize(img, resize)
            bboxes = self.face_detector.detect(
                img) if preprocessed_bboxes is None else preprocessed_bboxes[start_frame + i]
            if bboxes is None:
                print('No face is detected in frame %d' % (start_frame + i))
                continue
            x1, y1, x2, y2 = tuple(bboxes[0])
            face = img[y1:y2, x1:x2]
            mask = self.skin_detector.detect(face, is_dark=is_dark)
            V = face[mask != 0].reshape(-1, 3).astype(np.float64)
            C = np.matmul(V.T, V) / V.shape[0]
            eigval, eigvec = np.linalg.eigh(C)
            index = np.argsort(eigval)[::-1]
            # U[t, :, i] ith eigvec at t
            D[i], U[i] = eigval[index], eigvec[index]
            if display:
                tmp = cv2.bitwise_and(face, face, mask=mask)
                cv2.imshow('skin', tmp)
                if cv2.waitKey(1) != -1:
                    break
        if display:
            cv2.destroyWindow('skin')
        interval = int(fs * 32 / 20)
        sos = signal.butter(6, [0.7, 3.5], 'bandpass', fs=fs, output='sos')
        s = np.zeros(end_frame - start_frame, dtype=np.float64)
        for i in range(0, end_frame - start_frame - interval + 1, 1):
            SR = np.full((interval, 2), np.nan, dtype=np.float64)
            for j in range(interval):
                S = np.sqrt(D[i+j, 0] / D[i, 1:3])
                R = np.matmul(U[i+j, :, 0], U[i, :, 1:3]).reshape(-1)
                SR[j] = np.dot(U[i, :, 1:3], S * R)[:2]
            p = SR[:, 0] - (np.nanstd(SR[:, 0]) /
                            np.nanstd(SR[:, 1])) * SR[:, 1]
            if np.any(np.isnan(p)):
                if imputation:
                    valid = np.isfinite(p)
                    index = np.arange(len(p))
                    p = interp1d(index[valid], p[valid], kind='cubic',
                                 bounds_error=False, fill_value='extrapolate', axis=-1)(index)
                    if np.any(np.isnan(p)):  # all frames missing or missing without imputation)
                        p = np.zeros(len(p))
                else:
                    print('NaN in signal.')
                    return None
            s[i:i+interval] += (p - np.mean(p))
        sf = signal.sosfiltfilt(sos, s, padtype='odd')
        set_exact_frame(cap, icurr)
        return sf


class RPPGPOS:
    ''' 
    Color-based method for measuring blood volume pulse using Plane Orthogonal-to-Skin (POS).

    Reference:
        Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2016). Algorithmic principles of remote PPG. IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491.
    '''

    def __init__(self, face_detector=None, skin_detector=None):
        """
        Constructor.

        :param face_detector: FaceDetector instance (default: None).
        :type face_detector: FaceDetector, optional
        :param skin_detector: SkinDetector instance (default: None).
        :type skin_detector: SkinDetector, optional
        """
        self.face_detector = face_detector
        self.skin_detector = SkinDetector() if skin_detector is None else skin_detector

    def extract(self, cap, start_frame=0, end_frame=None, offset=0, rotate=None, resize=None, is_dark=False, imputation=False, display=False, preprocessed_bboxes=None):
        """
        Extracts the signal from the video.

        :param cap: VideoCapture object for the input video
        :type cap: cv2.VideoCapture
        :param start_frame: Starting frame index (default: 0).
        :type start_frame: int, optional
        :param end_frame: Ending frame index (default: None).
        :type end_frame: int, optional
        :param offset: Frame offset (default: 0).
        :type offset: int, optional
        :param rotate: Rotation flag for the input image (default: None).
        :type rotate: cv2.RotateFlags, optional
        :param resize: Resize shape for the input image (default: None).
        :type resize: tuple, optional
        :param is_dark: Whether the input image is dark (default: False).
        :type is_dark: bool, optional
        :param imputation: Whether to perform imputation for missing values (default: False).
        :type imputation: bool, optional
        :param display: Whether to display the image with ROI rectangles (default: False).
        :type display: bool, optional
        :param preprocessed_bboxes: Preprocessed bounding boxes (default: None).
        :type preprocessed_bboxes: list, optional
        :return: Processed signal
        :rtype: numpy.ndarray
        """
        if end_frame is None:
            # might be inaccurate
            end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if preprocessed_bboxes is None and self.face_detector is None:
            self.face_detector = FaceDetector()
        icurr = cap.get(cv2.CAP_PROP_POS_FRAMES)
        fs = cap.get(cv2.CAP_PROP_FPS)
        bgr = np.full((end_frame - start_frame, 3), np.nan, dtype=np.float64)
        set_exact_frame(cap, start_frame + offset)
        for i in range(end_frame - start_frame):
            ret, img = cap.read()
            if not ret:
                break
            if rotate is not None:
                img = cv2.rotate(img, rotate)
            if resize is not None:
                img = cv2.resize(img, resize)
            bboxes = self.face_detector.detect(
                img) if preprocessed_bboxes is None else preprocessed_bboxes[start_frame + i]
            if bboxes is None:
                print('No face is detected in frame %d' % (start_frame + i))
                continue
            x1, y1, x2, y2 = tuple(bboxes[0])
            face = img[y1:y2, x1:x2]
            mask = self.skin_detector.detect(face, is_dark=is_dark)
            bgr[i] = cv2.mean(face, mask=mask)[:3]
            if display:
                tmp = cv2.bitwise_and(face, face, mask=mask)
                cv2.imshow('skin', tmp)
                if cv2.waitKey(1) != -1:
                    break
        if display:
            cv2.destroyWindow('skin')
        if np.any(np.isnan(bgr)):
            if imputation:
                valid = np.isfinite(bgr)
                index = np.arange(len(bgr))
                bgr = interp1d(index[valid], bgr[valid], kind='cubic',
                               bounds_error=False, fill_value='extrapolate')(index)
            else:  # has missing value, but didn't deal with it
                print('NaN in signal.')
                return None
        interval = int(fs * 32 / 20)
        sos = signal.butter(6, [0.7, 3.5], 'bandpass', fs=fs, output='sos')
        wnd = signal.windows.hamming(interval)
        s = np.zeros(end_frame - start_frame, dtype=np.float64)
        for i in range(0, end_frame - start_frame - interval + 1, 1):
            bgrn = bgr[i:i+interval] / \
                np.mean(bgr[i:i+interval], axis=0, keepdims=True)
            xs = bgrn[:, 1] - bgrn[:, 0]  # Gn-2Bn
            ys = -2 * bgrn[:, 2] + bgrn[:, 1] + bgrn[:, 0]  # -2Rn + Gn + Bn
            alpha = np.std(xs) / np.std(ys)
            s[i:i+interval] += (xs + alpha * ys)
        sf = signal.sosfiltfilt(sos, s, padtype='odd')
        set_exact_frame(cap, icurr)
        return sf


class RPPGSPH:
    ''' 
    Color-based method for measuring blood volume pulse using spherical operator (SPH).

    Reference:
        Pilz, C. (2019). On the vector space in photoplethysmography imaging. In Proceedings of the IEEE International Conference on Computer Vision Workshops (pp. 0-0).
    '''

    def __init__(self, face_detector=None, skin_detector=None):
        """
        Constructor.

        :param face_detector: FaceDetector instance (default: None).
        :type face_detector: FaceDetector, optional
        :param skin_detector: SkinDetector instance (default: None).
        :type skin_detector: SkinDetector, optional
        """
        self.face_detector = face_detector
        self.skin_detector = SkinDetector() if skin_detector is None else skin_detector

    def extract(self, cap, start_frame=0, end_frame=None, offset=0, rotate=None, resize=None, is_dark=False, imputation=False, display=False, preprocessed_bboxes=None):
        """
        Extracts the signal from the video.

        :param cap: VideoCapture object for the input video
        :type cap: cv2.VideoCapture
        :param start_frame: Starting frame index (default: 0).
        :type start_frame: int, optional
        :param end_frame: Ending frame index (default: None).
        :type end_frame: int, optional
        :param offset: Frame offset (default: 0).
        :type offset: int, optional
        :param rotate: Rotation flag for the input image (default: None).
        :type rotate: cv2.RotateFlags, optional
        :param resize: Resize shape for the input image (default: None).
        :type resize: tuple, optional
        :param is_dark: Whether the input image is dark (default: False).
        :type is_dark: bool, optional
        :param imputation: Whether to perform imputation for missing values (default: False).
        :type imputation: bool, optional
        :param display: Whether to display the image with ROI rectangles (default: False).
        :type display: bool, optional
        :param preprocessed_bboxes: Preprocessed bounding boxes (default: None).
        :type preprocessed_bboxes: list, optional
        :return: Processed signal
        :rtype: numpy.ndarray
        """
        if end_frame is None:
            # might be inaccurate
            end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if preprocessed_bboxes is None and self.face_detector is None:
            self.face_detector = FaceDetector()
        icurr = cap.get(cv2.CAP_PROP_POS_FRAMES)
        fs = cap.get(cv2.CAP_PROP_FPS)
        s = np.full(end_frame - start_frame, np.nan, dtype=np.float64)
        set_exact_frame(cap, start_frame + offset)
        for i in range(end_frame - start_frame):
            ret, img = cap.read()
            if not ret:
                break
            if rotate is not None:
                img = cv2.rotate(img, rotate)
            if resize is not None:
                img = cv2.resize(img, resize)
            bboxes = self.face_detector.detect(
                img) if preprocessed_bboxes is None else preprocessed_bboxes[start_frame + i]
            if bboxes is None:
                print('No face is detected in frame %d' % (start_frame + i))
                continue
            x1, y1, x2, y2 = tuple(bboxes[0])
            face = img[y1:y2, x1:x2]
            mask = self.skin_detector.detect(face, is_dark=is_dark)
            pixels = face[mask != 0].astype(np.float64)
            sphere = pixels / np.linalg.norm(pixels, axis=1, keepdims=True)
            E = np.mean(sphere, axis=0)
            s[i] = np.arctan2(E[1], E[2])
            if display:
                tmp = cv2.bitwise_and(face, face, mask=mask)
                cv2.imshow('skin', tmp)
                if cv2.waitKey(1) != -1:
                    break
        if display:
            cv2.destroyWindow('skin')
        if np.any(np.isnan(s)):
            if imputation:
                valid = np.isfinite(s)
                index = np.arange(len(s))
                s = interp1d(index[valid], s[valid], kind='cubic',
                             bounds_error=False, fill_value='extrapolate')(index)
            else:  # has missing value, but didn't deal with it
                print('NaN in signal.')
                return None
        sos = signal.butter(6, [0.7, 3.5], 'bandpass', fs=fs, output='sos')
        sf = signal.sosfiltfilt(sos, s, padtype='odd')
        set_exact_frame(cap, icurr)
        return sf


class RPPGCSC:
    ''' 
    Color-based method for measuring blood volume pulse using Combination of simple chrominance signals (CSC).

    Reference:
        Wang, Chien-Chih (2020). Non-contact heart rate measurement based on facial videos. Master's Thesis, National Cheng Kung University, Tainan, Taiwan.

    '''

    def __init__(self, face_detector=None, skin_detector=None):
        """
        Constructor.

        :param face_detector: FaceDetector instance (default: None).
        :type face_detector: FaceDetector, optional
        :param skin_detector: SkinDetector instance (default: None).
        :type skin_detector: SkinDetector, optional
        """
        self.face_detector = face_detector
        self.skin_detector = SkinDetector() if skin_detector is None else skin_detector

    def extract(self, cap, start_frame=0, end_frame=None, offset=0, rotate=None, resize=None, is_dark=False, imputation=False, display=False, preprocessed_bboxes=None):
        """
        Extracts the signal from the video.

        :param cap: VideoCapture object for the input video
        :type cap: cv2.VideoCapture
        :param start_frame: Starting frame index (default: 0).
        :type start_frame: int, optional
        :param end_frame: Ending frame index (default: None).
        :type end_frame: int, optional
        :param offset: Frame offset (default: 0).
        :type offset: int, optional
        :param rotate: Rotation flag for the input image (default: None).
        :type rotate: cv2.RotateFlags, optional
        :param resize: Resize shape for the input image (default: None).
        :type resize: tuple, optional
        :param is_dark: Whether the input image is dark (default: False).
        :type is_dark: bool, optional
        :param imputation: Whether to perform imputation for missing values (default: False).
        :type imputation: bool, optional
        :param display: Whether to display the image with ROI rectangles (default: False).
        :type display: bool, optional
        :param preprocessed_bboxes: Preprocessed bounding boxes (default: None).
        :type preprocessed_bboxes: list, optional
        :return: Processed signal
        :rtype: numpy.ndarray
        """
        if end_frame is None:
            # might be inaccurate
            end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if preprocessed_bboxes is None and self.face_detector is None:
            self.face_detector = FaceDetector()
        icurr = cap.get(cv2.CAP_PROP_POS_FRAMES)
        fs = cap.get(cv2.CAP_PROP_FPS)
        bgr = np.full((end_frame - start_frame, 3), np.nan, dtype=np.float64)
        set_exact_frame(cap, start_frame + offset)
        for i in range(end_frame - start_frame):
            ret, img = cap.read()
            if not ret:
                break
            if rotate is not None:
                img = cv2.rotate(img, rotate)
            if resize is not None:
                img = cv2.resize(img, resize)
            bboxes = self.face_detector.detect(
                img) if preprocessed_bboxes is None else preprocessed_bboxes[start_frame + i]
            if bboxes is None:
                print('No face is detected in frame %d' % (start_frame + i))
                continue
            x1, y1, x2, y2 = tuple(bboxes[0])
            face = img[y1:y2, x1:x2]
            mask = self.skin_detector.detect(face, is_dark=is_dark)
            bgr[i] = cv2.mean(face, mask=mask)[:3]
            if display:
                tmp = cv2.bitwise_and(face, face, mask=mask)
                cv2.imshow('skin', tmp)
                if cv2.waitKey(1) != -1:
                    break
        if display:
            cv2.destroyWindow('skin')
        if np.any(np.isnan(bgr)):
            if imputation:
                valid = np.isfinite(bgr)
                index = np.arange(len(bgr))
                bgr = interp1d(index[valid], bgr[valid], kind='cubic',
                               bounds_error=False, fill_value='extrapolate')(index)
            else:  # has missing value, but didn't deal with it
                print('NaN in signal.')
                return None
        sos = signal.butter(6, 0.5, btype='lowpass', fs=fs, output='sos')
        bgrn = bgr / signal.sosfiltfilt(sos, bgr, axis=0, padtype='odd')
        x = bgrn[:, 1] - bgrn[:, 0]  # Gn - Bn
        y1 = bgrn[:, 1] - bgrn[:, 2]  # Gn - Rn
        y2 = bgrn[:, 0] - bgrn[:, 2]  # Bn - Rn
        interval = int(fs * 32 / 20) * 2
        beta = rolling_std(y1, window_len=interval, center=True) / \
            rolling_std(y2, window_len=interval, center=True)
        y = y1 + beta * y2
        alpha = rolling_std(x, window_len=interval, center=True) / \
            rolling_std(y, window_len=interval, center=True)
        s = x + alpha * y
        sos = signal.butter(6, [0.7, 3.5], 'bandpass', fs=fs, output='sos')
        sf = signal.sosfiltfilt(sos, s, padtype='odd')
        set_exact_frame(cap, icurr)
        return sf


class RPPGCHROMPRN:
    ''' 
    Color-based method for measuring blood volume pulse using CHROM model with PRNet (CHROM-PRN).

    Reference:
        Wang, Chien-Chih (2020). Non-contact heart rate measurement based on facial videos. Master's Thesis, National Cheng Kung University, Tainan, Taiwan.

    '''

    def __init__(self, face_detector=None, face_aligner=None, skin_detector=None):
        """
        Constructor.

        :param face_detector: FaceDetector instance (default: None).
        :type face_detector: FaceDetector, optional
        :param face_aligner: FaceAligner3D instance (default: None).
        :type face_aligner: FaceAligner3D, optional
        :param skin_detector: SkinDetector instance (default: None).
        :type skin_detector: SkinDetector, optional
        """
        self.face_detector = face_detector
        self.face_aligner = face_aligner  # FaceAligner3D
        self.skin_detector = SkinDetector() if skin_detector is None else skin_detector
        self.mask = [
            cv2.imread(os.path.join('model', 'mask_forehead.png'), 0),
            cv2.imread(os.path.join('model', 'mask_rightcheek.png'), 0),
            cv2.imread(os.path.join('model', 'mask_leftcheek.png'), 0)
        ]

    def extract(self, cap, start_frame=0, end_frame=None, offset=0, rotate=None, resize=None, is_dark=False, imputation=False, display=False, preprocessed_bboxes=None, preprocessed_colormap=None):
        """
        Extracts the signal from the video.

        :param cap: VideoCapture object for the input video
        :type cap: cv2.VideoCapture
        :param start_frame: Starting frame index (default: 0).
        :type start_frame: int, optional
        :param end_frame: Ending frame index (default: None).
        :type end_frame: int, optional
        :param offset: Frame offset (default: 0).
        :type offset: int, optional
        :param rotate: Rotation flag for the input image (default: None).
        :type rotate: cv2.RotateFlags, optional
        :param resize: Resize shape for the input image (default: None).
        :type resize: tuple, optional
        :param is_dark: Whether the input image is dark (default: False).
        :type is_dark: bool, optional
        :param imputation: Whether to perform imputation for missing values (default: False).
        :type imputation: bool, optional
        :param display: Whether to display the image with ROI rectangles (default: False).
        :type display: bool, optional
        :param preprocessed_bboxes: Preprocessed bounding boxes (default: None).
        :type preprocessed_bboxes: list, optional
        :return: Processed signal
        :rtype: numpy.ndarray
        """
        # i = offset (usually is i_triggered) for cap <=> i = 0 for preprocessed_bboxes and preprocessed_colormap (preprocessed_bboxes and preprocessed_colormap start from the triggered frame)
        if end_frame is None:
            # might be inaccurate
            end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if preprocessed_bboxes is None and preprocessed_colormap is None and self.face_detector is None:
            self.face_detector = FaceDetector()
        if preprocessed_colormap is None and self.face_aligner is None:
            self.face_aligner = FaceAligner3D()

        fs = cap.get(cv2.CAP_PROP_FPS)
        # store mean bgr for every frame (float64 to avoid numerical trouble while filtering)
        bgr = np.full((len(self.mask), end_frame-start_frame, 3),
                      np.nan, dtype=np.float64)
        set_exact_frame(cap, offset + start_frame)
        if preprocessed_colormap is None:
            icurr = cap.get(cv2.CAP_PROP_POS_FRAMES)
            set_exact_frame(cap, start_frame + offset)
        else:  # preprocessed_colormap is not None
            icurr = preprocessed_colormap.get(cv2.CAP_PROP_POS_FRAMES)
            set_exact_frame(preprocessed_colormap, start_frame)
        for i in range(end_frame - start_frame):
            if preprocessed_colormap is None:
                ret, img = cap.read()
                if not ret:
                    break
                if rotate is not None:
                    img = cv2.rotate(img, rotate)
                if resize is not None:
                    img = cv2.resize(img, resize)
                bboxes = self.face_detector.detect(
                    img) if preprocessed_bboxes is None else preprocessed_bboxes[start_frame + i]
                if bboxes is None:
                    print('No face is detected in frame %d' %
                          (start_frame + i))
                    continue
                bgcolor = np.array([0, 255, 0], dtype=np.uint8)
                colormap = self.face_aligner.align(
                    img, bboxes[0], bgcolor=bgcolor)
            else:
                ret, colormap = preprocessed_colormap.read()
                if not ret:
                    colormap = None
                    print('Failed to read frame %d from preprocessed colormap' % (
                        int(preprocessed_colormap.get(cv2.CAP_PROP_POS_FRAMES))))
            if colormap is None or np.all(colormap == bgcolor):
                print('No colormap in frame %d' % (start_frame + i))
                continue
            mask_skin = self.skin_detector.detect(colormap, is_dark=is_dark)
            if display:
                mask_display = np.zeros(mask_skin.shape, dtype=np.uint8)
            for j in range(len(self.mask)):
                mask = np.bitwise_and(mask_skin, self.mask[j])
                bgr[j, i] = cv2.mean(colormap, mask=mask)[:, 3] if np.any(
                    mask) else np.nan  # avoid divide by 0
                if display:
                    mask_display = np.bitwise_or(mask_display, self.mask[j])
            if display:
                mask_display = np.bitwise_and(mask_display, mask_skin)
                tmp = np.bitwise_and(colormap, colormap, mask=mask_display)
                cv2.imshow('skin', tmp)
                if cv2.waitKey(1) != -1:
                    break
        if display:
            cv2.destroyWindow('skin')

        if np.any(np.isnan(bgr)):
            if imputation:
                valid = np.isfinite(bgr)
                index = np.repeat(np.arange(len(end_frame - start_frame)), 3)
                index = np.tile(index, len(self.mask)).reshape(bgr.shape)
                bgr = interp1d(index[valid], bgr[valid], axis=1, kind='cubic',
                               bounds_error=False, fill_value='extrapolate')(index)
            else:
                print('NaN in signal.')
                return None

        interval = int(fs * 32 / 20)
        sos = signal.butter(6, [0.7, 3.5], 'bandpass', fs=fs, output='sos')
        wnd = signal.windows.hamming(interval)
        s = np.zeros((len(self.mask), end_frame -
                     start_frame), dtype=np.float64)
        bgrn = bgr / \
            rolling_mean(bgr, window_len=int(fs*2), axis=1, center=True)
        for i in range(0, end_frame - start_frame - interval + 1, interval // 2):
            bn, gn, rn = bgrn[:, i:i+interval, 0], bgrn[:,
                                                        i:i+interval, 1], bgrn[:, i:i+interval, 2]
            xs = 3 * rn - 2 * gn  # 3Rn - 2Gn
            ys = 1.5 * rn + gn - 1.5 * bn  # 1.5Rn + Gn - 1.5Bn
            xf = signal.sosfiltfilt(sos, xs, axis=-1, padtype='odd')
            yf = signal.sosfiltfilt(sos, ys, axis=-1, padtype='odd')
            alpha = np.std(xf, axis=-1, keepdims=True) / \
                np.std(yf, axis=-1, keepdims=True)
            s[:, i:i+interval] += (xf - alpha * yf) * wnd

        s = np.sum(s, axis=0)
        sf = signal.sosfiltfilt(sos, s, padtype='odd')
        if preprocessed_colormap is None:
            set_exact_frame(cap, icurr)
        else:  # preprocessed_colormap is not None
            set_exact_frame(preprocessed_colormap, icurr)
        return sf


class HYBRID:
    ''' 
    Color-based method for measuring blood volume pulse using GREEN, ICA, CHROM, POS, SPH, CSC, and CHROMPRN.

    Reference:
        - Verkruysse, W., Svaasand, L. O., & Nelson, J. S. (2008). Remote plethysmographic imaging using ambient light. Optics express, 16(26), 21434-21445.
        - Poh, M. Z., McDuff, D. J., & Picard, R. W. (2010). Non-contact, automated cardiac pulse measurements using video imaging and blind source separation. Optics express, 18(10), 10762-10774.
        - De Haan, G., & Jeanne, V. (2013). Robust pulse rate from chrominance-based rPPG. IEEE Transactions on Biomedical Engineering, 60(10), 2878-2886.
        - Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2016). Algorithmic principles of remote PPG. IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491.
        - Pilz, C. (2019). On the vector space in photoplethysmography imaging. In Proceedings of the IEEE International Conference on Computer Vision Workshops (pp. 0-0).
        - Wang, Chien-Chih (2020). Non-contact heart rate measurement based on facial videos. Master's Thesis, National Cheng Kung University, Tainan, Taiwan.

    '''

    def __init__(self, face_detector=None, face_aligner=None, skin_detector=None):
        """
        Constructor.

        :param face_detector: FaceDetector instance (default: None).
        :type face_detector: FaceDetector, optional
        :param face_aligner: FaceAligner3D instance (default: None).
        :type face_aligner: FaceAligner3D, optional
        :param skin_detector: SkinDetector instance (default: None).
        :type skin_detector: SkinDetector, optional
        """
        self.face_detector = face_detector
        self.face_aligner = face_aligner
        self.skin_detector = SkinDetector(
            backend=SkinDetector.BACKEND_YCRCB) if skin_detector is None else skin_detector
        self.ica_transformer = ICA(n_components=3, backend=ICA.BACKEND_JADE)
        # self.roi_green = np.array([ # xmin, ymin, xmax, ymax (x, y => [0, 1])
        #     [0.35, 0.22, 0.65, 0.30], # forehead (good)
        #     [0.49, 0.27, 0.51, 0.28], # forehead point (not good)
        #     [0.80, 0.20, 0.95, 0.45], # left temple (not good)
        #     [0.05, 0.05, 0.95, 0.95], # full face (better?)
        #     [0.66, 0.48, 0.90, 0.65], # [0.66, 0.45, 0.90, 0.60], # left cheek (not good)
        #     [0.10, 0.48, 0.34, 0.65], # [0.10, 0.45, 0.34, 0.60] # right cheek (not good)
        #     [0.30, 0.70, 0.70, 0.87]  # mouth (best? if static, currently this is more stable)
        # ], dtype=np.float32) # by left and right cheek is best?
        self.mask_chromprn = [
            cv2.imread(os.path.join('model', 'mask_forehead.png'), 0),
            cv2.imread(os.path.join('model', 'mask_rightcheek.png'), 0),
            cv2.imread(os.path.join('model', 'mask_leftcheek.png'), 0)
        ]

    def extract(self, cap, start_frame=0, end_frame=None, offset=0, rotate=None, resize=None, is_dark=False, imputation=False, display=False, preprocessed_bboxes=None, preprocessed_colormap=None):
        """
        Extracts the signal from the video.

        :param cap: VideoCapture object for the input video
        :type cap: cv2.VideoCapture
        :param start_frame: Starting frame index (default: 0).
        :type start_frame: int, optional
        :param end_frame: Ending frame index (default: None).
        :type end_frame: int, optional
        :param offset: Frame offset (default: 0).
        :type offset: int, optional
        :param rotate: Rotation flag for the input image (default: None).
        :type rotate: cv2.RotateFlags, optional
        :param resize: Resize shape for the input image (default: None).
        :type resize: tuple, optional
        :param is_dark: Whether the input image is dark (default: False).
        :type is_dark: bool, optional
        :param imputation: Whether to perform imputation for missing values (default: False).
        :type imputation: bool, optional
        :param display: Whether to display the image with ROI rectangles (default: False).
        :type display: bool, optional
        :param preprocessed_bboxes: Preprocessed bounding boxes (default: None).
        :type preprocessed_bboxes: list, optional
        :return: Processed signal
        :rtype: numpy.ndarray
        """
        if end_frame is None:
            # might be inaccurate
            end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if preprocessed_bboxes is None and self.face_detector is None:
            self.face_detector = FaceDetector()
        if preprocessed_colormap is None and self.face_aligner is None:
            self.face_aligner = FaceAligner3D()
        # start process
        fs = cap.get(cv2.CAP_PROP_FPS)
        bgr = np.full((end_frame - start_frame, 3), np.nan, dtype=np.float64)
        s_sph = np.full(end_frame - start_frame, np.nan, dtype=np.float64)
        # store mean bgr for every frame (float64 to avoid numerical trouble while filtering)
        bgr_mroi = np.full((len(self.mask_chromprn), end_frame -
                           start_frame, 3), np.nan, dtype=np.float64)
        # icurr = cap.get(cv2.CAP_PROP_POS_FRAMES)
        set_exact_frame(cap, offset + start_frame)
        if preprocessed_colormap is not None:
            set_exact_frame(preprocessed_colormap, start_frame)
        for i in range(end_frame - start_frame):
            # print(i)
            ret, img = cap.read()
            if not ret:
                break
            if rotate is not None:
                img = cv2.rotate(img, rotate)
            if resize is not None:
                img = cv2.resize(img, resize)
            if preprocessed_bboxes is None:
                bboxes = self.face_detector.detect(img)
                if bboxes is None:
                    print(f'No face is detected in frame {start_frame + i}')
                    continue
                bbox = bboxes[0]
            else:
                bbox = preprocessed_bboxes[start_frame + i, 1:]
            if preprocessed_colormap is None:
                bgcolor = np.array([0, 255, 0], dtype=np.uint8)
                colormap = self.face_aligner.align(
                    img, bboxes[0], bgcolor=bgcolor)
            else:
                ret, colormap = preprocessed_colormap.read()
                if not ret:
                    print(f'No preprocessed colormap in frame {start_frame+i}')
                    continue
            x1, y1, x2, y2 = tuple(bbox)
            face = img[y1:y2, x1:x2]
            try:
                mask_skin = self.skin_detector.detect(face, is_dark=is_dark)
            except:
                print(f'No face is detected in frame {start_frame + i}')
                continue
            bgr[i] = cv2.mean(face, mask=mask_skin)[:3]
            # sph
            pixels = face[mask_skin != 0].astype(np.float64)
            sphere = pixels / np.linalg.norm(pixels, axis=1, keepdims=True)
            E = np.mean(sphere, axis=0)
            s_sph[i] = np.arctan2(E[1], E[2])
            # chromprn
            mask_colormap = self.skin_detector.detect(
                colormap, is_dark=is_dark)
            if display:
                mask_display = mask_colormap.copy()
                for j in range(len(self.mask_chromprn)):
                    mask_display = np.bitwise_or(
                        mask_display, self.mask_chromprn[j])
            for j in range(len(self.mask_chromprn)):
                # self.mask_chromprn[j])
                mask = np.bitwise_and(mask_colormap, 1)
                bgr_mroi[j, i] = cv2.mean(colormap, mask=mask)[:3] if np.any(
                    mask) else np.full(3, np.nan)  # avoid divide by 0
            # # display
            if display:
                # normal
                skin = cv2.bitwise_and(face, face, mask=mask_skin)
                cv2.imshow('skin', skin)
                # chromprn
                tmp = np.bitwise_and(colormap, colormap, mask=mask_display)
                cv2.imshow('colormap skin', tmp)
                # update
                if cv2.waitKey(1) != -1:
                    break
        if display:
            cv2.destroyAllWindows()

        # imputation
        if np.any(np.isnan(bgr)):
            if imputation:
                valid = np.isfinite(bgr)
                index = np.arange(bgr.shape[0])
                index_valid = np.where(valid[:, 0])[0]
                bgr_valid = np.take_along_axis(
                    bgr, index_valid.reshape(-1, 1), axis=0)
                bgr = interp1d(index_valid, bgr_valid, axis=0, kind='cubic',
                               bounds_error=False, fill_value='extrapolate')(index)
            else:  # has missing value, but didn't deal with it
                print('NaN in signal bgr.')
                return None
        # haven't checked yet
        if np.any(np.isnan(s_sph)):
            if imputation:
                valid = np.isfinite(s_sph)
                index = np.arange(len(s_sph))
                s_sph = interp1d(index[valid], s_sph[valid], kind='cubic',
                                 bounds_error=False, fill_value='extrapolate')(index)
            else:  # has missing value, but didn't deal with it
                print('NaN in signal s_sph.')
                return None
        if np.any(np.isnan(bgr_mroi)):
            if imputation:
                try:
                    valid = np.isfinite(bgr_mroi)
                    index = np.arange(bgr_mroi.shape[1])
                    index_valid = np.where(valid[0, :, 0])[0]
                    bgr_mroi_valid = np.take_along_axis(
                        bgr_mroi, index_valid.reshape(1, -1, 1), axis=1)
                    bgr_mroi = interp1d(index_valid, bgr_mroi_valid, axis=1, kind='cubic',
                                        bounds_error=False, fill_value='extrapolate')(index)
                except:
                    print('NaN in signal bgr_mroi.')
                    return None
            else:
                print('NaN in signal bgr_mroi.')
                return None

        # for each method
        sos = signal.butter(6, [0.7, 3.5], 'bandpass', fs=fs, output='sos')

        # green (ok)
        s_green = signal.sosfiltfilt(sos, bgr[:, 1], padtype='odd', axis=-1)

        # ica (ok, similar)
        interval = int(fs * 128 / 20)
        wnd = signal.windows.hamming(interval)
        s_ica = np.zeros(end_frame - start_frame, dtype=np.float64)
        # identical to video-pulse
        bgrn = bgr / \
            rolling_mean(bgr, window_len=int(fs*2), axis=0, center=True)
        for i in range(0, end_frame - start_frame - interval + 1, interval // 2):
            x = self.ica_transformer.fit_transform(bgrn[i:i+interval])
            # identical to video-pulse
            xf = signal.sosfiltfilt(sos, x, padtype='odd', axis=0)
            snr = estimate_hr_snr_by_fft(xf, fs, axis=0)[
                1]  # different here, but similar
            s_ica[i:i+interval] += xf[:, np.argmax(snr)] * wnd
        s_ica = signal.sosfiltfilt(sos, s_ica, padtype='odd')

        # chrom (ok)
        interval = int(fs * 32 / 20)
        wnd = signal.windows.hamming(interval)
        s_chrom = np.zeros(end_frame - start_frame, dtype=np.float64)
        bgrn = bgr / \
            rolling_mean(bgr, window_len=int(fs*2), axis=0, center=True)
        for i in range(0, end_frame - start_frame - interval + 1, interval // 2):
            bn, gn, rn = bgrn[i:i+interval, 0], bgrn[i:i +
                                                     interval, 1], bgrn[i:i+interval, 2]
            xs = 3 * rn - 2 * gn  # 3Rn - 2Gn
            ys = 1.5 * rn + gn - 1.5 * bn  # 1.5Rn + Gn - 1.5Bn
            xf = signal.sosfiltfilt(sos, xs, padtype='odd')
            yf = signal.sosfiltfilt(sos, ys, padtype='odd')
            alpha = np.std(xf) / np.std(yf)
            s_chrom[i:i+interval] += (xf - alpha * yf) * wnd
        s_chrom = signal.sosfiltfilt(sos, s_chrom, padtype='odd')

        # pos
        interval = int(fs * 32 / 20)
        wnd = signal.windows.hamming(interval)
        s_pos = np.zeros(end_frame - start_frame, dtype=np.float64)
        for i in range(0, end_frame - start_frame - interval + 1, 1):
            bgrn = bgr[i:i+interval] / \
                np.mean(bgr[i:i+interval], axis=0, keepdims=True)
            xs = bgrn[:, 1] - bgrn[:, 0]  # Gn-2Bn
            ys = -2 * bgrn[:, 2] + bgrn[:, 1] + bgrn[:, 0]  # -2Rn + Gn + Bn
            alpha = np.std(xs) / np.std(ys)
            s_pos[i:i+interval] += (xs + alpha * ys)
        s_pos = signal.sosfiltfilt(sos, s_pos, padtype='odd')

        # sph (ok)
        s_sph = signal.sosfiltfilt(sos, s_sph, padtype='odd')

        # csc (ok)
        sos_lp = signal.butter(6, 0.5, btype='lowpass', fs=fs, output='sos')
        bgrn = bgr / signal.sosfiltfilt(sos_lp, bgr, axis=0, padtype='odd')
        x = bgrn[:, 1] - bgrn[:, 0]  # Gn - Bn
        y1 = bgrn[:, 1] - bgrn[:, 2]  # Gn - Rn
        y2 = bgrn[:, 0] - bgrn[:, 2]  # Bn - Rn
        interval = int(fs * 32 / 20) * 2
        beta = rolling_std(y1, window_len=interval, center=True) / \
            rolling_std(y2, window_len=interval, center=True)
        y = y1 + beta * y2
        alpha = rolling_std(x, window_len=interval, center=True) / \
            rolling_std(y, window_len=interval, center=True)
        s_csc = x + alpha * y
        s_csc = signal.sosfiltfilt(sos, s_csc, padtype='odd')

        # chromprn (ok, similar)
        interval = int(fs * 32 / 20)
        wnd = signal.windows.hamming(interval)
        s_chromprn = np.zeros(
            (len(self.mask_chromprn), end_frame - start_frame), dtype=np.float64)
        bgrn = bgr_mroi / \
            rolling_mean(bgr_mroi, window_len=int(fs*2), axis=1, center=True)
        for i in range(0, end_frame - start_frame - interval + 1, interval // 2):
            bn, gn, rn = bgrn[:, i:i+interval, 0], bgrn[:,
                                                        i:i+interval, 1], bgrn[:, i:i+interval, 2]
            xs = 3 * rn - 2 * gn  # 3Rn - 2Gn
            ys = 1.5 * rn + gn - 1.5 * bn  # 1.5Rn + Gn - 1.5Bn
            xf = signal.sosfiltfilt(sos, xs, axis=-1, padtype='odd')
            yf = signal.sosfiltfilt(sos, ys, axis=-1, padtype='odd')
            alpha = np.std(xf, axis=-1, keepdims=True) / \
                np.std(yf, axis=-1, keepdims=True)
            s_chromprn[:, i:i+interval] += (xf - alpha * yf) * wnd
        s_chromprn = np.sum(s_chromprn, axis=0)
        s_chromprn = signal.sosfiltfilt(sos, s_chromprn, padtype='odd')

        # set_exact_frame(cap, icurr) # just skip?
        return s_green, s_ica, s_chrom, s_pos, s_sph, s_csc, s_chromprn
