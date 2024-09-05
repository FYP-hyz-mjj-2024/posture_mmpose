import torch

alpha = 0.8
bbox_thr = 0.3
det_cat_id = 0
det_checkpoint = 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'
det_config = 'rtmdet_m_640-8xb32_coco-person.py'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
draw_bbox = False
draw_heatmap = False
input = 'demo.png'
kpt_thr = 0.3
nms_thr = 0.3
output_root = 'vis_results/'
pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth'
pose_config = 'td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py'
radius = 3
save_predictions = False
show = True
show_interval = 0
show_kpt_idx = False
skeleton_style = 'mmpose'
thickness = 1
