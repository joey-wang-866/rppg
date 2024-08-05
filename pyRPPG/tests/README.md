# Pipeline

### 1. Get bounding boxes and colormap from 3d face aligner using export_bbox_colormap.py

   Command: python3 export_bbox_colormap.py <list_json>

   Output: bounding boxes coordinates in npy and colormap video. Output path based on 'bbox' and 'colormap' in list json file
      

### 2. Get ground truth heart rate by ECG (gtp) and PPG (gt)

   Command: python3 export_hr_gt.py <list_json>

   Output: ground truth in npz file. Output path based on 'gt' in list json file
      

### 3. Get heart rate estimation for various rPPG methods

   Command: python3 export_hr_snr_rppg.py <list_json>

   Output: estimated heart rate in npz file. Output path based on 'hybrid' in list json file
      

### 4. Get statistics values (MAE, MAPE, SR, etc)

   Command: python3 export_statistics.py <list_json>

   Output: print statistics values

# Test codes

### 1. test_facedetector.py: test functionality of OpenCV face detector

   Command: python3 test_facedetector.py <video_file>

   Output: A popup window showing video with red bounding box following face movements.

### 2. test_facealigner3d.py: test functionality of PRNet 3D face aligner

   Command: python3 test_facealigner3d.py <video_file>

   Output: A popup window showing video with red bounding box and 3d face mask following face movements.

### 3. test_facealigner2d.py: test functionality of 2D face aligner

   Command: python3 test_facealigner2d.py <video_file>

   Output: A popup window showing video with red bounding box and face mask following face movements.

### 4. test_set_exact_frame.py: test set exact time frame of video

   Command: python3 test_set_exact_frame.py <video_file>

### 5. test_skindetector.py: test functionality of skin detector

   Command: python3 test_skindetector.py <video_file>

   Output: A popup window showing black and white video where white color is skin.
