import os
import mmcv
import torch
import cv2
import os.path as osp
import numpy as np
from mmseg.apis.inference import init_segmentor, visual_result, inference_segmentor
from mmseg.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
import time
from tqdm import tqdm
import rasterio

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':


    # __import__("pudb").set_trace()
    config_file = '/data/projects/mmsegmentation-0.16.0/configs/deeplabv3/deeplabv3_r50-d8_512x512_80k_flood_data_test.py'
    checkpoint_file = '/data/projects/mmsegmentation-0.16.0/work_dirs/deeplabv3_r50-d8_512x512_80k_flood_data/iter_40000.pth'   # checkpoint_file = '/data/projects/mmdetection-1.1.0/work_dirs/bankcard_6_11/epoch_40.pth'

    segmentor = init_segmentor(config_file, checkpoint_file)

    feature_path = '/data/dataset/flood_data/Floodwater/flood-train-images/'
    filenames = os.listdir(feature_path)

    filenames = [name.split('_')[0] for name in filenames]
    filenames = set(filenames)

    visual_reslut = '/data/dataset/flood_data/Floodwater/visual/'

    if not os.path.exists(visual_reslut):
        os.mkdir(visual_reslut)

    for img_name in tqdm(filenames):

        with rasterio.open(os.path.join(feature_path, img_name + '_vv.tif')) as vv:
            vv_path = vv.read(1)
        with rasterio.open(os.path.join(feature_path, img_name + '_vh.tif')) as vh:
            vh_path = vh.read(1)

        img = np.stack([vv_path, vh_path], axis=-1)
        # Min-max normalization
        min_norm = -77
        max_norm = 26
        img = np.clip(img, min_norm, max_norm)
        img = (img - min_norm) / (max_norm - min_norm)

        result = inference_segmentor(segmentor, img)
        res_img = visual_result(segmentor, result, palette=[[0,0, 0], [225, 225, 225]])
        # res_img = res_img * 255
        cv2.imwrite(os.path.join(visual_reslut, img_name+'.jpg'), res_img)
