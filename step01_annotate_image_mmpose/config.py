import torch

alpha = 0.8
bbox_thr = 0.3              # Confidence, Higher = Fewer boxes
bbox_thr_single = 0.85      # Confidence for detecting a single person.
det_cat_id = 0
det_checkpoint = '../model_config/checkpoints/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth'
det_config = '../model_config/configs/rtmdet_nano_320-8xb32_coco-person.py'
# device = 'cuda:0' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
draw_bbox = False
draw_heatmap = False
input = 'demo.png'
kpt_thr = 0.3
nms_thr = 0.3
output_root = 'vis_results/'
pose_checkpoint = '../model_config/checkpoints/rtmpose-tiny_simcc-aic-coco_pt-aic-coco_420e-256x192-cfc8f33d_20230126.pth'
pose_config = '../model_config/configs/rtmpose-t_8xb256-420e_coco-256x192.py'
radius = 3
save_predictions = False
show = True
show_interval = 0
show_kpt_idx = False
skeleton_style = 'mmpose'
thickness = 1
