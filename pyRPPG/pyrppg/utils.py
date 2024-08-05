import os
import wget
import bz2
import gdown
import cv2
import dlib
import numpy as np
import bottleneck as bn
from .prnet import PosPrediction
from sklearn.decomposition import FastICA
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html
# def interp1d_along_axis():


def set_exact_frame(cap, target_frame):
    """
    For some encoding, if using FFMPEG to decode, it cannot set to the exact frame.
    It might set to the keyframe instead, so if this happens, we just read from the beginning until the exact frame.

    :param cap: VideoCapture object from OpenCV.
    :type cap: cv2.VideoCapture
    :param target_frame: The target frame number to set.
    :type target_frame: int
    """
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    for i in range(target_frame, -fps, -fps):
        if i < 0:
            i = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) == i:
            for j in range(i, target_frame):
                cap.read()
            break
    assert int(cap.get(cv2.CAP_PROP_POS_FRAMES)
               ) == target_frame, 'Failed to set the exact frame position'


def rolling_mean(x, window_len, center=False, axis=-1):
    """
    Calculate rolling mean of the input array.

    :param x: Input array.
    :type x: numpy.ndarray
    :param window_len: Window length for rolling mean.
    :type window_len: int
    :param center: If True, calculate centered rolling mean (default: False).
    :type center: bool, optional
    :param axis: Axis along which the rolling mean is computed (default: -1).
    :type axis: int, optional
    :return: The result array with rolling mean values.
    :rtype: float64
    """
    assert window_len > 1, 'Window length should be greater than 1'
    x = x.astype(np.float64)
    if center:
        n1 = window_len // 2  # divide into two parts (n//2) and (n-n//2)
        pad_width = np.zeros((x.ndim, 2), dtype=np.int32)
        # pad to end, so the mean it calculates is centered
        pad_width[axis, 1] = n1
        x_padded = np.pad(x, pad_width, mode='constant',
                          constant_values=np.nan)
        res = bn.move_mean(x_padded, window_len, min_count=1, axis=axis)
        return np.take(res, np.arange(n1, res.shape[axis]), axis=axis)
    else:
        return bn.move_mean(x, window_len, min_count=1, axis=axis)
        # mean from [i-window_len+1, ..., i-1, i]


def rolling_std(x, window_len, center=False, axis=-1):
    """
    Calculate rolling standard deviation of the input array.

    :param x: Input array.
    :type x: numpy.ndarray
    :param window_len: Window length for rolling standard deviation.
    :type window_len: int
    :param center: If True, calculate centered rolling standard deviation (default: False).
    :type center: bool, optional
    :param axis: Axis along which the rolling standard deviation is computed (default: -1).
    :type axis: int, optional
    :return: The result array with rolling standard deviation values.
    :rtype: float64
    """
    assert window_len > 1, 'Window length should be greater than 1'
    x = x.astype(np.float64)
    if center:
        n1 = window_len // 2  # divide into two parts (n//2) and (n-n//2)
        pad_width = np.zeros((x.ndim, 2), dtype=np.int32)
        # pad to end, so the mean it calculates is centered
        pad_width[axis, 1] = n1
        x_padded = np.pad(x, pad_width, mode='constant',
                          constant_values=np.nan)
        res = bn.move_std(x_padded, window_len, min_count=1, axis=axis)
        return np.take(res, np.arange(n1, res.shape[axis]), axis=axis)
    else:
        return bn.move_std(x, window_len, min_count=1, axis=axis)


def rolling_max(x, window_len, center=False, axis=-1):
    """
    Calculate rolling maximum of the input array.

    :param x: Input array.
    :type x: numpy.ndarray
    :param window_len: Window length for rolling maximum.
    :type window_len: int
    :param center: If True, calculate centered rolling maximum (default: False).
    :type center: bool, optional
    :param axis: Axis along which the rolling maximum is computed (default: -1).
    :type axis: int, optional
    :return: The result array with rolling maximum values.
    :rtype: float64
    """
    assert window_len > 1, 'Window length should greater than 1'
    dtype = x.dtype
    x = x.astype(np.float64)
    if center:
        n1 = window_len // 2  # divide in to two part (n//2) and (n-n//2)
        pad_width = np.zeros((x.ndim, 2), dtype=np.int32)
        # pad to end, so the mean it calculate is centered
        pad_width[axis, 1] = n1
        x_padded = np.pad(x, pad_width, mode='constant',
                          constant_values=np.nan)
        res = bn.move_max(x_padded, window_len, min_count=1, axis=axis)
        return np.take(res, np.arange(n1, res.shape[axis]), axis=axis).astype(dtype)
    else:
        return bn.move_max(x, window_len, min_count=1, axis=axis).astype(dtype)


def rolling_min(x, window_len, center=False, axis=-1):
    """
    Calculate rolling minimum of the input array.

    :param x: Input array.
    :type x: numpy.ndarray
    :param window_len: Window length for rolling minimum.
    :type window_len: int
    :param center: If True, calculate centered rolling minimum (default: False).
    :type center: bool, optional
    :param axis: Axis along which the rolling minimum is computed (default: -1).
    :type axis: int, optional
    :return: The result array with rolling minimum values.
    :rtype: float64
    """
    assert window_len > 1, 'Window length should greater than 1'
    dtype = x.dtype
    x = x.astype(np.float64)
    if center:
        n1 = window_len // 2  # divide in to two part (n//2) and (n-n//2)
        pad_width = np.zeros((x.ndim, 2), dtype=np.int32)
        # pad to end, so the mean it calculate is centered
        pad_width[axis, 1] = n1
        x_padded = np.pad(x, pad_width, mode='constant',
                          constant_values=np.nan)
        res = bn.move_min(x_padded, window_len, min_count=1, axis=axis)
        return np.take(res, np.arange(n1, res.shape[axis]), axis=axis).astype(dtype)
    else:
        return bn.move_min(x, window_len, min_count=1, axis=axis).astype(dtype)


def rolling_argmax(x, window_len, center=False, axis=-1):
    """
    Calculate rolling argmax of the input array.

    :param x: Input array.
    :type x: numpy.ndarray
    :param window_len: Window length for rolling argmax.
    :type window_len: int
    :param center: If True, calculate centered rolling argmax (default: False).
    :type center: bool, optional
    :param axis: Axis along which the rolling argmax is computed (default: -1).
    :type axis: int, optional
    :return: The result array with rolling argmax values.
    :rtype: float64
    """
    assert window_len > 1, 'Window length should greater than 1'
    x = x.astype(np.float64)
    if center:
        n1 = window_len // 2  # divide in to two part (n//2) and (n-n//2)
        pad_width = np.zeros((x.ndim, 2), dtype=np.int32)
        # pad to end, so the mean it calculate is centered
        pad_width[axis, 1] = n1
        x_padded = np.pad(x, pad_width, mode='constant',
                          constant_values=np.nan)
        shape = np.ones(x.ndim, dtype=np.int32)
        shape[axis] = x_padded.shape[axis]
        index = np.arange(x_padded.shape[axis]).reshape(shape)
        tmp = bn.move_argmax(x_padded, window_len, min_count=1,
                             axis=axis).astype(np.int32)
        res = index - tmp
        return np.take(res, np.arange(n1, res.shape[axis]), axis=axis)
    else:
        shape = np.ones(x.ndim, dtype=np.int32)
        shape[axis] = x.shape[axis]
        index = np.arange(x.shape[axis]).reshape(shape)
        amax = bn.move_argmax(x, window_len, min_count=1,
                              axis=axis).astype(np.int32)
        # argmax from [i-window_len+1, ..., i-1, i], index 0 for right most in the window and increase from right to left [..., 2, 1, 0]
        return index - amax


def rolling_argmin(x, window_len, center=False, axis=-1):
    """
    Calculate rolling argmin of the input array.

    :param x: Input array.
    :type x: numpy.ndarray
    :param window_len: Window length for rolling argmin.
    :type window_len: int
    :param center: If True, calculate centered rolling argmin (default: False).
    :type center: bool, optional
    :param axis: Axis along which the rolling argmin is computed (default: -1).
    :type axis: int, optional
    :return: The result array with rolling argmin values.
    :rtype: float64
    """
    assert window_len > 1, 'Window length should greater than 1'
    x = x.astype(np.float64)
    if center:
        n1 = window_len // 2  # divide in to two part (n//2) and (n-n//2)
        pad_width = np.zeros((x.ndim, 2), dtype=np.int32)
        # pad to end, so the mean it calculate is centered
        pad_width[axis, 1] = n1
        x_padded = np.pad(x, pad_width, mode='constant',
                          constant_values=np.nan)
        shape = np.ones(x.ndim, dtype=np.int32)
        shape[axis] = x_padded.shape[axis]
        index = np.arange(x_padded.shape[axis]).reshape(shape)
        tmp = bn.move_argmin(x_padded, window_len, min_count=1,
                             axis=axis).astype(np.int32)
        res = index - tmp
        return np.take(res, np.arange(n1, res.shape[axis]), axis=axis)
    else:
        shape = np.ones(x.ndim, dtype=np.int32)
        shape[axis] = x.shape[axis]
        index = np.arange(x.shape[axis]).reshape(shape)
        amin = bn.move_argmin(x, window_len, min_count=1,
                              axis=axis).astype(np.int32)
        # argmin from [i-window_len+1, ..., i-1, i], index 0 for right most in the window and increase from right to left [..., 2, 1, 0]
        return index - amin


def url_download(url, path, gdrive=False):
    """
    Download a file and create directory if necessary.

    :param url: URL of the file to download.
    :type url: string
    :param path: Local path to save the downloaded file.
    :type path: string
    :param gdrive: If True, use gdown for downloading from Google Drive (default: False).
    :type gdrive: bool, optional
    """
    dir, fname = os.path.split(path)
    if not os.path.isdir(dir):
        os.makedirs(dir)
    if not gdrive:
        wget.download(url, path)
    else:
        gdown.download(url, path, quiet=False)


def rect_iou(rect1, rect2):
    """
    Calculate intersection over union for two rectangles.
    Rectangle in [x, y, w, h] format.

    :param rect1: First rectangle.
    :type rect1: numpy.ndarray
    :param rect2: Second rectangle.
    :type rect2: numpy.ndarray
    :return: Intersection over union value.
    :rtype: float
    """
    l1, r1 = rect1[0], rect1[0] + rect1[2]
    l2, r2 = rect2[0], rect2[0] + rect2[2]
    w = 0 if r1 < l2 or r2 < l1 else (min(r1, r2) - max(l1, l2))
    t1, b1 = rect1[1], rect1[1] + rect1[3]
    t2, b2 = rect2[1], rect2[1] + rect2[3]
    h = 0 if b1 < t2 or b2 < t1 else (min(b1, b2) - max(t1, t2))
    return (w * h) / (rect1[2] * rect1[3] + rect2[2] * rect2[3] - w * h)


def combined_rect(rect1, rect2):
    """
    Calculate the combined rectangle from two input rectangles.
    Rectangle in [x, y, w, h] format.

    :param rect1: First rectangle.
    :type rect1: numpy.ndarray
    :param rect2: Second rectangle.
    :type rect2: numpy.ndarray
    :return: Combined rectangle.
    :rtype: numpy.ndarray
    """
    return np.array([min(rect1[0], rect2[0]),
                     min(rect1[1], rect2[1]),
                     max(rect1[0]+rect1[2], rect2[0]+rect2[2]),
                     max(rect1[1]+rect1[3], rect2[1]+rect2[3])])


class JADE:
    """
    JADE (Joint Approximate Diagonalization of Eigenmatrices) implementation for Independent Component Analysis.

    Reference:
    Rutledge, D. N., & Bouveresse, D. J. R. (2013). Independent components analysis with the JADE algorithm.
    TrAC Trends in Analytical Chemistry, 50, 22-32.

    Original MATLAB implementation:
    https://github.com/sccn/eeglab/blob/develop/functions/sigprocfunc/jader.m
    """

    def __init__(self, n_components=None):
        """
        Initialize JADE object.

        :param n_components: Number of components to extract, default is None which means extract all components.
        :type n_components: int, optional
        """
        self.n_components = n_components
        self.components = None  # demixing
        self.mixing = None
        self.mean = None

    def fit(self, X):
        """
        Fit the model with X. X: n_samples * n_sensors.

        :param X: Data matrix with shape (n_samples, n_sensors).
        :type X: numpy.ndarray
        """
        n_samples, n_sensors = X.shape
        if self.n_components is None:
            self.n_components = n_sensors
        assert self.n_components <= n_sensors, 'n_components should be less or equal to n_sensors'
        num_sources = self.n_components

        # Zero mean
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # Whitening and project onto signal subspace
        eigval, eigvec = np.linalg.eig(np.matmul(X.T, X))  # PCA
        top_indices = np.argsort(
            eigval)[:-num_sources-1:-1]  # m most significant
        scale, eigvec_top = np.sqrt(
            eigval[top_indices]), eigvec[:, top_indices]

        # Whitener (top m eigvec with inverse scale (normalize))
        W = (1 / scale) * eigvec_top
        Winv = (scale * eigvec_top).T  # Pseudo inverse of whitener
        X = np.matmul(X, W)

        # Compute cumulant matrices
        # Only half size needed because of real symmetry
        num_cumul = (num_sources * (num_sources + 1)) // 2
        CM = np.zeros((num_sources, num_sources * num_cumul))
        Y = X.T
        I = np.eye(num_sources)
        r = np.arange(num_sources, dtype=np.int32)
        for i in range(num_sources):
            Qij = np.matmul((Y[i] * Y[i] / n_samples) * Y, X) - I
            Qij[i, i] -= 2
            CM[:, r] = Qij
            r += num_sources
            for j in range(i):
                Qij = np.matmul((Y[i] * Y[j] / n_samples) * Y, X)
                Qij[i, j] -= 1
                Qij[j, i] -= 1
                CM[:, r] = np.sqrt(2) * Qij
                r += num_sources

        # Joint diagonalization of the cumulant matrices
        # Initialization
        V = np.linalg.eig(CM[:, :num_sources])[1]  # eigvec
        for i in range(0, num_sources * num_cumul, num_sources):
            CM[:, i:(i + num_sources)] = np.matmul(CM[:, i:(i + num_sources)], V)

        CM = np.matmul(V.T, CM)

        # Parameters
        # A statistically significant threshold
        threshold = 0.01 / np.sqrt(n_samples)
        encore, sweep, updates = True, 0, 0
        while encore:
            encore = False
            sweep += 1
            for i in range(num_sources - 1):
                for j in range(i + 1, num_sources):
                    ii = np.arange(i, num_sources * num_cumul,
                                   num_sources, dtype=np.int32)
                    jj = np.arange(j, num_sources * num_cumul,
                                   num_sources, dtype=np.int32)

                    # Computation of Givens angle
                    g = np.vstack(
                        [CM[i, ii] - CM[j, jj], CM[i, jj] + CM[j, ii]])
                    gg = np.matmul(g, g.T)
                    ton, toff = gg[0, 0] - gg[1, 1], gg[0, 1] + gg[1, 0]
                    theta = 0.5 * \
                        np.arctan2(
                            toff, ton + np.sqrt(ton * ton + toff * toff))

                    # Givens update
                    if np.abs(theta) > threshold:
                        encore = True
                        updates = updates + 1
                        c, s = np.cos(theta), np.sin(theta)
                        G = np.array([[c, -s], [s, c]])
                        V[:, (i, j)] = np.matmul(V[:, (i, j)], G)
                        CM[(i, j), :] = np.matmul(G.T, CM[(i, j), :])
                        a = c * CM[:, ii] + s * CM[:, jj]
                        b = -s * CM[:, ii] + c * CM[:, jj]
                        CM[:, ii] = a
                        CM[:, jj] = b

        # A separating matrix
        B = np.matmul(W, V)  # nxm (transpose of original B)
        # Sort components (most energetic at first)
        A = np.matmul(V.T, Winv)  # mxn (transpose of original A)
        keys = np.argsort(np.sum(A * A, axis=1))[::-1]
        B, A = B[:, keys], A[keys, :]

        # Signs are fixed by forcing the first row of B
        signs = np.sign(np.sign(B[0]) + 0.1)
        B, A = signs * B, signs.reshape(-1, 1) * A
        self.components = B.T
        self.mixing = A.T

    def transform(self, X):
        """
        Apply the dimensionality reduction on X.

        :param X: Data matrix with shape (n_samples, n_sensors).
        :type X: numpy.ndarray
        :return: Transformed data matrix.
        :rtype: numpy.ndarray
        """
        return np.matmul(X - self.mean, self.components.T)

    def fit_transform(self, X):
        """
        Fit the model with X and apply the dimensionality reduction on X.

        :param X: Data matrix with shape (n_samples, n_sensors).
        :type X: numpy.ndarray
        :return: Transformed data matrix.
        :rtype: numpy.ndarray
        """
        self.fit(X)
        return self.transform(X)


class ICA:
    """
    Independent Component Analysis.

    Attributes:
        - BACKEND_FASTICA: Integer constant for FastICA backend.
        - BACKEND_JADE: Integer constant for JADE backend.
        - BACKEND_INFOMAX: Integer constant for Infomax backend.
        - BACKEND_SOBI: Integer constant for SOBI backend.
    """
    BACKEND_FASTICA = 0
    BACKEND_JADE = 1
    BACKEND_INFOMAX = 2
    BACKEND_SOBI = 3

    def __init__(self, n_components, backend=BACKEND_JADE):
        """
        Initialize the ICA class.

        :param n_components: Number of components to extract.
        :type n_components: int
        :param backend: Backend for ICA algorithm (default: JADE).
        :type backend: int, optional
        """
        self.backend = backend
        if backend == ICA.BACKEND_FASTICA:
            self.transformer = FastICA(
                n_components=n_components, random_state=0)
        elif backend == ICA.BACKEND_JADE:
            self.transformer = JADE(n_components=n_components)
        elif backend == ICA.BACKEND_INFOMAX:
            self.transformer = None  # Not implemented yet
        elif backend == ICA.BACKEND_SOBI:
            self.transformer = None  # Not implemented yet
        else:
            raise ValueError('Unknown backend.')

    def fit(self, X):
        """
        Fit the ICA model with the given data.

        :param X: Data matrix with shape (n_samples, n_sensors).
        :type X: numpy.ndarray
        """
        self.transformer.fit(X)

    def transform(self, X):
        """
        Apply the dimensionality reduction on X.

        :param X: Data matrix with shape (n_samples, n_sensors).
        :type X: numpy.ndarray
        :return: Transformed data matrix.
        :rtype: numpy.ndarray
        """
        return self.transformer.transform(X)

    def fit_transform(self, X):
        """
        Fit the ICA model with X and apply the dimensionality reduction on X.

        :param X: Data matrix with shape (n_samples, n_sensors).
        :type X: numpy.ndarray
        :return: Transformed data matrix.
        :rtype: numpy.ndarray
        """
        return self.transformer.fit_transform(X)


class FaceDetector:
    """
    Face detector with multiple backend options.

    Attributes:
        - BACKEND_CV2_SSD: OpenCV pretrained SSD framework using ResNet-10 like architecture as a backbone.
        - BACKEND_CV2_HAAR: OpenCV pretrained Cascaded HAAR classifiers (ensemble front face and profile face model).
        - BACKEND_CV2_LBF: OpenCV pretrained Cascaded LBF classifiers (ensemble front face and profile face model).
        - BACKEND_DLIB_HOG: Dlib pretrained HOG classifier.
        - BACKEND_DLIB_MMOD: Dlib pretrained MMOD classifier.
    """
    BACKEND_CV2_SSD = 0
    BACKEND_CV2_HAAR = 1
    BACKEND_CV2_LBF = 2
    BACKEND_DLIB_HOG = 3
    BACKEND_DLIB_MMOD = 4

    def __init__(self, backend=BACKEND_CV2_SSD, download=True, model_path=None, config_path=None, threshold=None):
        """
        Initialize the FaceDetector class.

        :param backend: Backend for face detection (default: CV2_SSD).
        :type backend: int, optional
        :param download: Whether to download model files if not found (default: True).
        :type download: bool, optional
        :param model_path: Path to the model file (default: None).
        :type model_path: str, optional
        :param config_path: Path to the config file (default: None).
        :type config_path: str, optional
        :param threshold: Detection threshold (default: None).
        :type threshold: float, optional
        """
        self.backend = backend
        if backend == FaceDetector.BACKEND_CV2_SSD:
            if model_path is None:
                model_path = os.path.join(
                    'model', 'res10_300x300_ssd_iter_140000.caffemodel')
            if config_path is None:
                config_path = os.path.join('model', 'deploy.prototxt')

            # Download model and config files if needed
            self._prepare_files(download, model_path, config_path)

            self.model = cv2.dnn.readNetFromCaffe(config_path, model_path)
            self.threshold = 0.5 if threshold is None else threshold

        elif backend == FaceDetector.BACKEND_CV2_HAAR or backend == FaceDetector.BACKEND_CV2_LBF:
            if backend == FaceDetector.BACKEND_CV2_HAAR:
                model_name = 'haarcascade_frontalface_default.xml'
                config_name = 'haarcascade_profileface.xml'
            else:
                model_name = 'lbpcascade_frontalface_improved.xml'
                config_name = 'lbpcascade_profileface.xml'

            if model_path is None:
                model_path = os.path.join('model', model_name)
            if config_path is None:
                config_path = os.path.join('model', config_name)

            # Download model and config files if needed
            self._prepare_files(download, model_path, config_path)

            self.model1 = cv2.CascadeClassifier(model_path)
            self.model2 = cv2.CascadeClassifier(config_path)

        elif backend == FaceDetector.BACKEND_DLIB_HOG:
            self.model = dlib.get_frontal_face_detector()

        elif backend == FaceDetector.BACKEND_DLIB_MMOD:
            if model_path is None:
                model_path = os.path.join(
                    'model', 'mmod_human_face_detector.dat')

            # Download model file if needed
            self._prepare_files(download, model_path)

            self.model = dlib.cnn_face_detection_model_v1(model_path)

        else:
            raise ValueError('Unknown backend.')

    def detect(self, img):
        """
        Detect face in a color image (BGR, 8UC3) and return bounding boxes [(x_left, y_top, x_right, y_bottom), ...].
        Some models are not orientation invariant, so you should watch out for the orientation by yourself.

        :param img: Input image in BGR format.
        :type img: numpy.ndarray
        :return: Bounding boxes of detected faces.
        :rtype: numpy.ndarray
        """
        bboxes = None
        if self.backend == FaceDetector.BACKEND_CV2_SSD:
            blob = cv2.dnn.blobFromImage(
                img, 1.0, (300, 300), (104, 117, 123), swapRB=False)
            # TODO: because resize to 300x300, if frame size is large, maybe divide into several overlapped imgs and detect is better?
            self.model.setInput(blob)
            output = self.model.forward().squeeze()  # 1x1x200x7 -> 200x7
            face_detected = output[:, 2] > self.threshold
            if np.any(face_detected):
                h, w, ch = img.shape
                dims = np.array([w-1, h-1, w-1, h-1])
                bboxes = (output[face_detected, 3:7] * dims).astype(np.int32)
        elif self.backend == FaceDetector.BACKEND_CV2_HAAR or self.backend == FaceDetector.BACKEND_CV2_LBF:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces1 = self.model1.detectMultiScale(
                gray, scaleFactor=1.25, minNeighbors=3, minSize=(img.shape[1]//5, img.shape[0]//5))
            faces2 = self.model2.detectMultiScale(
                gray, scaleFactor=1.25, minNeighbors=3, minSize=(img.shape[1]//5, img.shape[0]//5))
            if len(faces1) > 0 and len(faces2) > 0:  # ensemble the results
                faces = []  # xywh
                for i in range(len(faces1)):
                    for j in range(i+1, len(faces2)):
                        if rect_iou(faces1[i], faces2[j]) > 0.3:
                            faces.append(combined_rect(faces1[i], faces2[j]))
                        else:
                            faces.append(faces1[i])
                            faces.append(faces2[j])
                bboxes = np.array(faces, dtype=np.int32).reshape(-1, 4)
                # bboxes = np.vstack((faces1, faces2)).astype(np.int32)
            elif len(faces1) > 0:
                bboxes = faces1.astype(np.int32)
            elif len(faces2) > 0:
                bboxes = faces2.astype(np.int32)
            if bboxes is not None:
                bboxes[:, 2:] += bboxes[:, :2]
        elif self.backend == FaceDetector.BACKEND_DLIB_HOG or self.backend == FaceDetector.BACKEND_DLIB_MMOD:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            output = self.model(rgb, 0)  # 0 times upsampling
            if len(output) > 0:
                if self.backend == FaceDetector.BACKEND_DLIB_HOG:
                    bboxes = [(d.left(), d.top(), d.right(), d.bottom())
                              for d in output]
                else:
                    bboxes = [(d.rect.left(), d.rect.top(),
                               d.rect.right(), d.rect.bottom()) for d in output]
                bboxes = np.array(bboxes, dtype=np.int32)
        # clip
        if bboxes is not None:
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img.shape[1]-1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img.shape[0]-1)
        return bboxes

    # def detect_batch(self, imgs):
    #     '''Detect face in multiple images together.
    #     For backend using GPU, the performance might be better.
    #     For backend using CPU, the performance is same.
    #     '''
    #     bboxes = [None] * len(imgs)
    #     return bboxes

    def _prepare_files(self, download, model_path, config_path=None):
        """
        Download model and config files if needed.
        :param download: Whether to download files if not found.
        :type download: bool
        :param model_path: Path to the model file.
        :type model_path: str
        :param config_path: Path to the config file (default: None).
        :type config_path: str
        """
        if not os.path.exists(model_path):
            assert download, 'Error: model_path does not exist.'
            self._download_file(model_path)

        if config_path is not None and not os.path.exists(config_path):
            assert download, 'Error: config_path does not exist.'
            self._download_file(config_path)

    def _download_file(self, file_path):
        """
        Download file from the corresponding URL.

        :param file_path: Path to save the downloaded file.
        :type file_path: str
        """
        dir, file_name = os.path.split(file_path)
        if not os.path.isdir(dir):
            os.makedirs(dir)

        url = self._get_url_for_file(file_name)
        if url:
            url_download(url, file_path)
        else:
            raise ValueError(f'Unable to find URL for file: {file_name}')

    def _get_url_for_file(self, file_name):
        """
        Get the download URL for the given file name.

        :param file_name: Name of the file to download.
        :type file_name: str
        :return: URL for the file.
        :rtype: str
        """
        urls = {
            'res10_300x300_ssd_iter_140000.caffemodel': 'https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel',
            'deploy.prototxt': 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt',
            'haarcascade_frontalface_default.xml': 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml',
            'haarcascade_profileface.xml': 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_profileface.xml',
            'lbpcascade_frontalface_improved.xml': 'https://raw.githubusercontent.com/opencv/opencv/master/data/lbpcascades/lbpcascade_frontalface_improved.xml',
            'lbpcascade_profileface.xml': 'https://raw.githubusercontent.com/opencv/opencv/master/data/lbpcascades/lbpcascade_profileface.xml',
            'mmod_human_face_detector.dat.bz2': 'http://dlib.net/files/mmod_human_face_detector.dat.bz2',
        }

        return urls.get(file_name)


class FaceAligner2D:
    """
    2D Face Aligner with multiple backend options.

    Attributes:
        - BACKEND_CV_KAZEMI: OpenCV Facemark pretrained KAZEMI.
        - BACKEND_DLIB_KAZEMI: Dlib pretrained KAZEMI.
    """
    BACKEND_CV_KAZEMI = 0   # OpenCV Facemark pretrained KAZEMI
    BACKEND_DLIB_KAZEMI = 1  # Dlib pretrained KAZEMI

    def __init__(self, backend=BACKEND_DLIB_KAZEMI, download=True, model_path=None):
        """
        Initialize the FaceAligner2D with the given backend and model.

        :param backend: Backend to use for face alignment (default: FaceAligner2D.BACKEND_DLIB_KAZEMI).
        :type backend: int
        :param download: Whether to download the model file if not found (default: True).
        :type download: bool, optional
        :param model_path: Path to the model file (default: None).
        :type model_path: str, optional
        """
        self.backend = backend
        if backend == FaceAligner2D.BACKEND_CV_KAZEMI:
            if model_path is None:
                model_path = os.path.join('model', 'lbfmodel.yaml')
            self._check_and_download_model(download, model_path)
            self.model = cv2.face.createFacemarkLBF()
            self.model.loadModel(model_path)
        elif backend == FaceAligner2D.BACKEND_DLIB_KAZEMI:
            if model_path is None:
                model_path = os.path.join(
                    'model', 'shape_predictor_68_face_landmarks.dat')
            self._check_and_download_model(download, model_path, is_bz2=True)
            self.model = dlib.shape_predictor(model_path)
        else:
            raise ValueError('Unknown backend.')

    def align(self, img, bbox):
        """
        Align the face in the given image using the bounding box.

        :param img: Input image in BGR format.
        :type img: numpy.ndarray
        :param bbox: Bounding box of the face (x_left, y_top, x_right, y_bottom).
        :type bbox: tuple
        :return: 68x2 numpy array of landmark coordinates, or None if the bounding box is None.
        :rtype: numpy.ndarray
        """
        if bbox is None:
            return None
        coords = None
        if self.backend == FaceAligner2D.BACKEND_CV_KAZEMI:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            x_left, y_top, x_right, y_bottom = tuple(bbox)
            roi_array = np.array(
                [[x_left, y_top, x_right - x_left, y_bottom - y_top]], dtype=np.int32)
            ret, tmp = self.model.fit(gray, roi_array)
            coords = tmp[0][0] if ret else None
        elif self.backend == FaceAligner2D.BACKEND_DLIB_KAZEMI:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            x_left, y_top, x_right, y_bottom = tuple(bbox)
            tmp = self.model(gray, dlib.rectangle(
                x_left, y_top, x_right, y_bottom))
            coords = np.array([(tmp.part(i).x, tmp.part(i).y)
                              for i in range(68)], dtype=np.int32)
        return coords

    def _check_and_download_model(self, download, model_path, is_bz2=False):
        """
        Check if the model file exists, and download it if not found and download is set to True.

        :param download: Whether to download the model file if not found.
        :type download: bool
        :param model_path: Path to the model file.
        :type model_path: str
        :param is_bz2: Whether the model file is compressed with bz2 (default: False).
        :type is_bz2: bool, optional

        """
        if not os.path.exists(model_path):
            assert download, 'Error: model_path does not exist.'
            if is_bz2:
                tmp_path = model_path + '.bz2'
                url_download(self._get_url_for_file(
                    os.path.basename(tmp_path)), tmp_path)
                with open(tmp_path, 'rb') as src, open(model_path, 'wb') as dst:
                    dst.write(bz2.decompress(src.read()))
                os.remove(tmp_path)
            else:
                url_download(self._get_url_for_file(
                    os.path.basename(model_path)), model_path)

    def _get_url_for_file(self, file_name):
        """
        Get the download URL for the given file name.

        :param file_name: Name of the file to download.
        :type file_name: str
        :return: URL for the file.
        :rtype: str
        """
        urls = {
            'lbfmodel.yaml': 'https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml',
            'shape_predictor_68_face_landmarks.dat.bz2': 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
        }

        return urls.get(file_name)


class FaceAligner3D:
    """
    3D Face Aligner with PRNet backend.

    Attributes:
        - BACKEND_PRN: PRNet backend.
    """
    BACKEND_PRN = 0

    def __init__(self, backend=BACKEND_PRN, download=True, model_path=None, config_path=None, face_index_path=None, face_mesh_path=None):
        """
        Initialize the FaceAligner3D with the given backend and model.

        :param backend: Backend to use for face alignment (default: FaceAligner3D.BACKEND_PRN).
        :type backend: int
        :param download: Whether to download the model files if not found (default: True).
        :type download: bool, optional
        :param model_path: Path to the model file (default: None).
        :type model_path: str, optional
        :param config_path: Path to the config file (default: None).
        :type config_path: str, optional
        :param face_index_path: Path to the face index file (default: None).
        :type face_index_path: str, optional
        :param face_mesh_path: Path to the face mesh file (default: None).
        :type face_mesh_path: str, optional
        """
        self.backend = backend
        if backend == FaceAligner3D.BACKEND_PRN:
            if model_path is None:
                model_path = os.path.join(
                    'model', '256_256_resfcn256_weight.data-00000-of-00001')
            if config_path is None:
                config_path = os.path.join(
                    'model', '256_256_resfcn256_weight.index')
            if face_index_path is None:
                face_index_path = os.path.join(
                    'model', 'face_ind.txt')  # start from 1?
            if face_mesh_path is None:
                face_mesh_path = os.path.join(
                    'model', 'triangles.txt')  # start from 1?
            if not os.path.exists(model_path):
                assert download, 'Error: model_path does not exist.'
                url_download(
                    'https://drive.google.com/u/0/uc?export=download&confirm=sXXA&id=1UoE-XuW1SDLUjZmJPkIZ1MLxvQFgmTFH', model_path, gdrive=True)
            if not os.path.exists(config_path):
                assert download, 'Error: config_path does not exist.'
                url_download(
                    'https://github.com/YadiraF/PRNet/raw/master/Data/net-data/256_256_resfcn256_weight.index', config_path)
            if not os.path.exists(face_index_path):
                assert download, 'Error: face_index_path does not exist.'
                url_download(
                    'https://github.com/YadiraF/PRNet/raw/master/Data/uv-data/face_ind.txt', face_index_path)
            if not os.path.exists(face_mesh_path):
                assert download, 'Error: face_mesh_path does not exist.'
                url_download(
                    'https://github.com/YadiraF/PRNet/raw/master/Data/uv-data/triangles.txt', face_mesh_path)
            self.model = PosPrediction(256, 256)
            # load model file through config_path
            self.model.restore(os.path.splitext(config_path)[0])
            self.face_index = np.loadtxt(face_index_path).astype(np.int32)
            self.face_mesh = np.loadtxt(face_mesh_path).astype(np.int32)
            self.morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        else:
            raise ValueError('Unknown backend.')

    def align(self, img, bbox, bg_color=(0, 0, 0), projection=False):
        """
        Align the given image with the specified bounding box.

        :param img: Image to be aligned.
        :type img: numpy.ndarray
        :param bbox: Bounding box for alignment.
        :type bbox: tuple
        :param bg_color: Background color (default: (0, 0, 0)).
        :type bg_color: tuple, optional
        :param projection: Whether to return projection (default: False).
        :type projection: bool, optional
        :return: Aligned image, and projection if requested.
        :rtype: numpy.ndarray, (numpy.ndarray, optional)
        """
        if bbox is None:
            return None
        color_map = None
        if self.backend == FaceAligner3D.BACKEND_PRN:
            # expand bounding box and crop
            x1, y1, x2, y2 = tuple(bbox)
            l = 1.58 * 0.25 * (np.abs(x2 - x1) + np.abs(y2 - y1))
            x0, y0 = 0.5 * (x1 + x2) - l, 0.5 * (y1 + y2) - \
                l  # upper left anchor of new box
            s = 128 / l  # ratio of output and original size
            # img_to_input (col major for OpenCV)
            M = np.array([[s, 0, -s * x0], [0, s, -s * y0],
                         [0, 0, 1]], dtype=np.float32)
            # output_to_img (row major Numpy)
            M_inv = np.array([[1/s, 0, 0], [0, 1/s, 0],
                              [x0, y0, 1]], dtype=np.float32)
            input_img = img.astype(np.float32) * (1/255)  # normalize
            input_img = cv2.warpAffine(input_img, M[:2, :], (256, 256))  # crop
            input_img = cv2.cvtColor(
                input_img, cv2.COLOR_BGR2RGB)  # PRNet use RGB (tf)

            # get position map
            coords = self.model.predict(
                input_img).reshape(-1, 3)  # output is float32
            z = coords[:, 2] * (1/s)  # extract z and scale to original size
            coords[:, 2] = 1.0  # construct homogeneous coordinates
            coords = np.matmul(coords, M_inv)  # to image coordinates
            coords[:, 2] = z  # put z back
            pos_map = coords.reshape(256, 256, 3)  # position map (float32)

            # get visibility of facial vertices (only back culling)
            coords_face = coords[self.face_index]
            visibility = np.zeros(256 * 256, dtype=np.uint8)
            visibility[self.face_index] = 255
            normal = np.cross(  # normal vector of mesh
                coords_face[self.face_mesh[:, 1]] - \
                coords_face[self.face_mesh[:, 0]],
                coords_face[self.face_mesh[:, 2]] - coords_face[self.face_mesh[:, 0]])
            # if any neighbor mesh is back
            is_back = self.face_mesh[normal[:, 2] < -1e-3].flatten()
            visibility[self.face_index[is_back]] = 0

            # upsampling, morphology
            mask = cv2.resize(visibility.reshape(256, 256),
                              (512, 512), interpolation=cv2.INTER_NEAREST)
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_DILATE, self.morph, iterations=2)
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE, self.morph, iterations=2)
            # default interpolation in cv2.resize is bicubic (float32)
            pos_map = cv2.resize(pos_map, (512, 512))
            color_map = cv2.remap(img, pos_map[:, :, :2], None, interpolation=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))  # smoother due to interpolation
            color_map[mask == 0] = bg_color  # set to bg_color

            # output
            if projection:
                projection_img = img.copy()
                x = np.clip(coords_face[:, 0].astype(
                    np.int32), 0, projection_img.shape[1] - 1)
                y = np.clip(coords_face[:, 1].astype(
                    np.int32), 0, projection_img.shape[0] - 1)
                projection_img[y[::4], x[::4]] = np.array(
                    [0, 0, 255], dtype=np.uint8)  # 1/4 density
                return color_map, projection_img
            else:
                return color_map


class SkinDetector:
    """
    A class that provides a skin detector using different color space models.

    The module uses OpenCV to convert the input image to different color spaces
    and extract skin pixels based on predefined thresholds.

    References:
        - Basilio, J. A. M., Torres, G. A., Pérez, G. S., Medina, L. K. T., & Meana, H. M. P. (2011). Explicit image detection using YCbCr space color model as skin detection. Applications of Mathematics and Computer Engineering, 123-128.
        - Gomez, G., & Morales, E. (2002, July). Automatic feature construction and a simple rule induction algorithm for skin detection. In Proc. of the ICML workshop on Machine Learning in Computer Vision (Vol. 31).
        - Kwon, O. Y., & Chien, S. I. (2018). Adaptive Skin Color Detection through Iterative Illuminant Color Estimation and Conversion for Preferred Skin Color Reproduction. Molecular Crystals and Liquid Crystals, 677(1), 105-117.

    Attributes:
        - BACKEND_YCRCB: An integer constant to represent the YCbCr color space model backend.
        - BACKEND_RCA: An integer constant to represent the RCA color space model backend.
        - BACKEND_XYZ: An integer constant to represent the XYZ color space model backend.
        - BACKEND_YES: An integer constant to represent the YES color space model backend.
        - BACKEND_HLS: An integer constant to represent the HLS color space model backend.
    """
    # Basilio, J. A. M., Torres, G. A., Pérez, G. S., Medina, L. K. T., & Meana, H. M. P. (2011). Explicit image detection using YCbCr space color model as skin detection. Applications of Mathematics and Computer Engineering, 123-128.
    BACKEND_YCRCB = 0
    # Gomez, G., & Morales, E. (2002, July). Automatic feature construction and a simple rule induction algorithm for skin detection. In Proc. of the ICML workshop on Machine Learning in Computer Vision (Vol. 31).
    BACKEND_RCA = 1
    BACKEND_XYZ = 2  # proposed
    # Kwon, O. Y., & Chien, S. I. (2018). Adaptive Skin Color Detection through Iterative Illuminant Color Estimation and Conversion for Preferred Skin Color Reproduction. Molecular Crystals and Liquid Crystals, 677(1), 105-117.
    BACKEND_YES = 3
    BACKEND_HLS = 4  # similar to YCrCb

    def __init__(self, backend=BACKEND_YCRCB):
        """
        Initializes a new SkinDetector object with the specified backend.

        :param backend: an integer specifying the color space and algorithm to use for skin detection.
        :type backend: int
        :raises ValueError: if the specified backend is not a valid value.
        """
        self.backend = backend
        if backend == SkinDetector.BACKEND_YCRCB:
            # # original threshold
            # self.lowerb = 81, 134, 78 # lower bound of YCrCb (inRange is >= and <=)
            # self.upperb = 255, 172, 126 # upper bound of YCrCb
            self.lowerb = 0.4, 0.5, 0.3  # YCrCb (float version)
            self.upperb = 1.0, 0.7, 0.5
            self.lowerb_dark = 0.05, 0.5, 0.3  # YCrCb (float version)
            self.upperb_dark = 0.8, 0.7, 0.5
        elif backend == SkinDetector.BACKEND_RCA:
            pass
        elif backend == SkinDetector.BACKEND_XYZ:
            pass
        elif backend == SkinDetector.BACKEND_YES:
            pass
        elif backend == SkinDetector.BACKEND_HLS:
            pass
        else:
            raise ValueError('Unknown backend.')

    def detect(self, img, is_dark=False):  # 8UC3
        """
        Detects skin pixels in the specified image using the current backend and threshold values.

        :param img: an input image in BGR format.
        :type img: numpy.ndarray
        :param is_dark: a bool value indicating whether to use the dark skin thresholds or not (default: False).
        :type is_dark: bool, optional
        :return: a binary mask of the same size as the input image with white pixels indicating skin pixels and black pixels indicating non-skin pixels.
        :rtype: numpy.ndarray
        """
        mask = None
        if self.backend == SkinDetector.BACKEND_YCRCB:
            ycrcb = cv2.cvtColor(img.astype(np.float32) *
                                 (1/255), cv2.COLOR_BGR2YCrCb)
            if is_dark:
                mask = cv2.inRange(ycrcb, self.lowerb_dark, self.upperb_dark)
            else:
                mask = cv2.inRange(ycrcb, self.lowerb, self.upperb)
        elif self.backend == SkinDetector.BACKEND_XYZ:
            xyz = cv2.cvtColor(img.astype(np.float32) *
                               (1/255), cv2.COLOR_BGR2XYZ)
            x, y, z = xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2]
            mask1 = y > (1.8 * z - x)  # good, especially for cloth
            # key for green background and yellow, (Need to adjust)
            mask2 = y < x + 0.05
            mask3 = y > 0.15  # shadow surround landmark, hair
            mask4 = np.logical_and(z < 1.1, z > 0.15)  # remove dark part
            mask = np.logical_and.reduce(
                (mask1, mask2, mask3, mask4)).astype(np.uint8) * 255
        elif self.backend == SkinDetector.BACKEND_HLS:
            hls = cv2.cvtColor(img.astype(np.float32) *
                               (1/255), cv2.COLOR_BGR2HLS)
            h, l, s = hls[:, :, 0], hls[:, :, 1], hls[:, :, 2]
            mask1 = np.logical_and(h > 10, h < 70)
            mask2 = l > 0.2
            mask3 = np.square(1-s) + np.square(0.8-l) < 1.2
            mask = np.logical_and.reduce((mask1, mask2)).astype(np.uint8) * 255
        elif self.backend == SkinDetector.BACKEND_YES:
            pass
        elif self.backend == SkinDetector.BACKEND_RCA:
            bgr = img.astype(np.float32) * (1/255)
            b, g, r = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]
            x0 = r + g + b  # move all denominator to right to avoid divided by zero
            x1 = r + g > 0.685 * x0
            x2 = r - g > 0.049 * x0
            x3 = g * b > 0.067 * x0 * x0
            x4 = b < 1.249 * g
            x5 = g < 0.324 * x0
            mask = np.logical_and.reduce(
                (x1, x2, x3, x4, x5)).astype(np.uint8) * 255
        return mask
