import torch

# Boundary Box Detection Settings
bbox_thr = 0.3              # Confidence for detecting multiple-people.
bbox_thr_single = 0.85      # Confidence for detecting a single person.
det_cat_id = 0              # Detection category ID

# Model Configurations: Bbox detection + Pose estimation
# Real-time - Tiny
det_checkpoint = './model_config/checkpoints/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth'
det_config = './model_config/configs/rtmdet_nano_320-8xb32_coco-person.py'
pose_checkpoint = './model_config/checkpoints/rtmpose-tiny_simcc-aic-coco_pt-aic-coco_420e-256x192-cfc8f33d_20230126.pth'
pose_config = './model_config/configs/rtmpose-t_8xb256-420e_coco-256x192.py'
# Training - Medium
det_checkpoint_train = '../model_config/checkpoints/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth'
det_config_train = '../model_config/configs/rtmdet_nano_320-8xb32_coco-person.py'
pose_checkpoint_train = '../model_config/checkpoints/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth'
pose_config_train = '../model_config/configs/rtmpose-m_8xb256-420e_coco-256x192.py'

# Hardware Settings
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Visualizer Settings
alpha = 0.8                 # Opacity of drawn
draw_bbox = True
draw_heatmap = False
input = 'demo.png'
kpt_thr = 0.3               # Visualizing keypoint thresholds
nms_thr = 0.3               # Visualizing bboxes thresholds of IoU of Bboxes
output_root = 'vis_results/'
radius = 3
save_predictions = False
show = True
show_interval = 0
show_kpt_idx = False
skeleton_style = 'mmpose'
thickness = 1
