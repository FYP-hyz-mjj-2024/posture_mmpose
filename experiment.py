import argparse

from mmcv import imread
from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples


def load_default_args():
    args = argparse.Namespace()
    args.alpha = 0.8
    args.bbox_thr = 0.3
    args.det_cat_id = 0
    args.det_checkpoint = 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'
    args.det_config = 'mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py'
    args.device = 'cuda:0'
    args.draw_bbox = False
    args.draw_heatmap = False
    args.input = input
    args.kpt_thr = 0.3
    args.nms_thr = 0.3
    args.output_root = 'vis_results/'
    args.out_file = 'data/demo/demo_output.jpg'
    args.pose_checkpoint = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth'
    args.pose_config = '../configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py'
    args.radius = 3
    args.save_predictions = False
    args.show = True
    args.show_interval = 0
    args.show_kpt_idx = False
    args.skeleton_style = 'mmpose'
    args.thickness = 1
    return args


register_all_modules()
args = load_default_args()

# Initialize Model
config_file = 'model_config/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
checkpoint_file = 'model_config/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
model = init_model(config_file, checkpoint_file, device='cuda:0')  # or device='cuda:0'

# Initialize Visualizer
model.cfg.visualizer.radius = args.radius
model.cfg.visualizer.alpha = args.alpha
model.cfg.visualizer.line_width = args.thickness

visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.set_dataset_meta(model.dataset_meta, skeleton_style=args.skeleton_style)

# Inference
# results = inference_topdown(model, 'demo.jpg')
batch_results = inference_topdown(model, 'demo.jpg')
results = merge_data_samples(batch_results)
print(results)

# Show results
img = imread('data/demo/demo.jpg', channel_order='rgb')
visualizer.add_datasample(
        'result',
        img,
        data_sample=results,
        draw_gt=False,
        draw_bbox=True,
        kpt_thr=args.kpt_thr,
        draw_heatmap=args.draw_heatmap,
        show_kpt_idx=args.show_kpt_idx,
        skeleton_style=args.skeleton_style,
        show=args.show,
        out_file=args.out_file)


