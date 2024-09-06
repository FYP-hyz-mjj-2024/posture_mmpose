## Configure Project

### Step 0. Conda
&emsp; Please make sure that you are under the conda environment. If you are not, please do the following in the 
anaconda prompt terminal:

#### 0.1 Create Virtual Environment
```shell
conda create --prefix <PATH_TO_YOUR_VENV_ROOT_FOLDER> python=3.8 -y
```
#### 0.2 Activate Virtual Environment
```shell
conda activate <PATH_TO_YOUR_VENV_ROOT_FOLDER>
```
#### 0.3 Go to Project Dir
```shell
cd <PATH_TO_YOUR_CLONED_PROJECT>
```

### Step 1. Install Pytorch
We have inspected that `mmcv` does not work with pytorch with a higher version. Under a higher torch version, `cuda:0`
is not available, eventhough `torch.cuda.is_available()` returns `True`.

According to this issue: https://github.com/open-mmlab/mmdetection/issues/11530#issuecomment-1996359661, 
`mmcv` only works with pytorch with version `2.1.0`. And it has been confirmed by us. Please run:

```shell
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

You should have these installed:

| Package            | Build                   |
|--------------------|-------------------------|
| pytorch-2.1.0      | py3.8_cuda11.8_cudnn8_0 |
| torchaudio-2.1.0   | py38_cu118              |
| torchvision-0.16.0 | py38_cu118              |

### 2. Install MM Packages

#### 2.1 Openmim
For windows users, please install openmim by running:
```shell
<PATH_TO_YOUR_VIRTUAL_ENVIRONMENT>/Scripts/pip.exe install -U openmim
```
This is done to ensure that you have used the correct pip. Otherwise, the package will be installed to the base 
environment, which is not cool.

For mac users, please install openmim by running:
```shell
<PATH_TO_YOUR_VIRTUAL_ENVIRONMENT>/bin/pip install -U openmim
```
after you have activated your conda environment.

#### 2.2 MM Packages
There are four MM packages you need to install. Please install the EXACT version listed in the form below. This is the
best solution we could get to prevent package conflicts. For more information, please visit 
https://mmcv.readthedocs.io/en/latest/get_started/installation.html.

| Package  | Version | Source                                    |
|----------|---------|-------------------------------------------|
| mmcv     | 2.1.0   | https://github.com/open-mmlab/mmcv        |
| mmdet    | 3.2.0   | https://github.com/open-mmlab/mmdetection |
| mmengine | 0.10.4  | https://github.com/open-mmlab/mmengine    |
| mmpose   | 1.3.2   | https://github.com/open-mmlab/mmpose      |

Install MM related packages:
```shell
mim install "mmcv==2.1.0" "mmdet==3.2.0" "mmengine==0.10.4" "mmpose==1.3.2"
```

#### 2.3 Checkpoint & Configuration Files
&emsp; Checkpoint files are essential to this project, yet they are too big to upload to github. 
There are two kinds of files: Checkpoint files and config files for both boundary detection and pose estimation.
Please download all of them by clicking these links:

- config files (`.py`):
  - det config (boundary detection): https://github.com/open-mmlab/mmpose/blob/dev-1.x/projects/rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py
  - pose config (pose estimation): https://github.com/open-mmlab/mmpose/blob/dev-1.x/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-t_8xb256-420e_coco-256x192.py
After downloading from the browser, please move them into `model_config/configs/`. You may need to create this
folder first.

- checkpoint files (`.pth`):
  - det checkpoint: https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth
  - pose checkpoint: https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-tiny_simcc-aic-coco_pt-aic-coco_420e-256x192-cfc8f33d_20230126.pth
After downloading from the browser, please move them into `model_config/checkpoints/`. You may need to create this
folder first.

## Name of Key Points
Below is a table of the categories and details of key-points, according to the opensource project mmpose. We named
the categories and details. The name of each keypoint will be `<CATEGORY>-<DETAILS>`.

For instance, for keypoint `Right_eye` in category `Body` at index `2`, the name of this keypoint
is restricted to be `Body-Right_eye`.

| Index | Category | Details               |
|-------|----------|-----------------------|
| 0     | Body     | Chin                  |
| 1     | Body     | Left_eye              |
| 2     | Body     | Right_eye             |
| 3     | Body     | Left_ear              |
| 4     | Body     | Right_ear             |
| 5     | Body     | Left_shoulder         |
| 6     | Body     | Right_shoulder        |
| 7     | Body     | Left_elbow            |
| 8     | Body     | Right_elbow           |
| 9     | Body     | Left_wrist            |
| 10    | Body     | Right_wrist           |
| 11    | Body     | Left_hip              |
| 12    | Body     | Right_hip             |
| 13    | Body     | Left_knee             |
| 14    | Body     | Right_knee            |
| 15    | Body     | Left_ankle            |
| 16    | Body     | Right_ankle           |
| 17    | Foot     | Left_toe              |
| 18    | Foot     | Left_pinky            |
| 19    | Foot     | Left_heel             |
| 20    | Foot     | Right_toe             |
| 21    | Foot     | Right_pinky           |
| 22    | Foot     | Right_heel            |
| 23    | Face     | Right_hairroot        |
| 24    | Face     | Right_zyngo           |
| 25    | Face     | Right_face_top        |
| 26    | Face     | Right_face_mid        |
| 27    | Face     | Right_face_bottom     |
| 28    | Face     | Right_chin_top        |
| 29    | Face     | Right_chin_mid        |
| 30    | Face     | Right_chin_bottom     |
| 31    | Face     | Chin                  |
| 32    | Face     | Left_chin_bottom      |
| 33    | Face     | Left_chin_mid         |
| 34    | Face     | Left_chin_top         |
| 35    | Face     | Left_face_bottom      |
| 36    | Face     | Left_face_mid         |
| 37    | Face     | Left_face_top         |
| 38    | Face     | Left_zyngo            |
| 39    | Face     | Left_hairroot         |
| 40    | Face     | Right_eyebrow_out     |
| 41    | Face     | Right_eyebrow_out_mid |
| 42    | Face     | Right_eyebrow_mid     |
| 43    | Face     | Right_eyebrow_mid_in  |
| 44    | Face     | Right_eyebrow_in      |
| 45    | Face     | Left_eyebrow_in       |
| 46    | Face     | Left_eyebrow_mid_in   |
| 47    | Face     | Left_eyebrow_mid      |
| 48    | Face     | Left_eyebrow_out_mid  |
| 49    | Face     | Left_eyebrow_out      |
| 50    | Face     | Nose_top              |
| 51    | Face     | Nose_top_mid          |
| 52    | Face     | Nose_bottom_mid       |
| 53    | Face     | Nose_bottom           |
| 54    | Face     | Right_nostril_out     |
| 55    | Face     | Right_nostril_mid     |
| 56    | Face     | Nostril               |
| 57    | Face     | Left_nostril_mid      |
| 58    | Face     | Left_nostril_out      |
| 59    | Face     | Right_eye_out         |
| 60    | Face     | Right_eye_up_out      |
| 61    | Face     | Right_eye_up_in       |
| 62    | Face     | Right_eye_in          |
| 63    | Face     | Right_eye_down_in     |
| 64    | Face     | Right_eye_down_out    |
| 65    | Face     | Left_eye_in           |
| 66    | Face     | Left_eye_up_in        |
| 67    | Face     | Left_eye_up_out       |
| 68    | Face     | Left_eye_out          |
| 69    | Face     | Left_eye_down_out     |
| 70    | Face     | Left_eye_down_in      |
| 71    | Face     | Lips_l1_right_out     |
| 72    | Face     | Lips_l1_right_mid     |
| 73    | Face     | Lips_l1_right_in      |
| 74    | Face     | Lips_l1_mid           |
| 75    | Face     | Lips_l1_left_in       |
| 76    | Face     | Lips_l1_left_mid      |
| 77    | Face     | Lips_l1_left_out      |
| 78    | Face     | Lips_l4_left_out      |
| 79    | Face     | Lips_l4_left_in       |
| 80    | Face     | Lips_l4_mid           |
| 81    | Face     | Lips_l4_right_in      |
| 82    | Face     | Lips_l4_right_out     |
| 83    | Face     | Lips_l2_right_out     |
| 84    | Face     | Lips_l2_right_in      |
| 85    | Face     | Lips_l2_mid           |
| 86    | Face     | Lips_l2_left_in       |
| 87    | Face     | Lips_l2_left_out      |
| 88    | Face     | Lips_l3_left          |
| 89    | Face     | Lips_l3_mid           |
| 90    | Face     | Lips_l3_right         |
