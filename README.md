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

## Body Points