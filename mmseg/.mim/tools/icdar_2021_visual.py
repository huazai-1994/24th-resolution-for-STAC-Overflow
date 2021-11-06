import os
import mmcv
import torch
import cv2
import os.path as osp
import numpy as np
from inference import init_segmentor, visual_result, inference_segmentor
# from mmdet.apis.inference import (init_detector, LoadImage)
from mmdet.apis.inference import LoadImage
from mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
import pycocotools.mask as maskUtils
import time
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':



    config_file = './configs/htc/htc_without_semantic_r50_fpn_1x_icdar.py'
    checkpoint_file = "/data/projects/mmdetection_icadr2021/work_dirs/idcard_2_5/epoch_99.pth"   # checkpoint_file = '/data/projects/mmdetection-1.1.0/work_dirs/bankcard_6_11/epoch_40.pth'

    segmentor = init_segmentor(config_file, checkpoint_file)

    filename_vv = './data/test_features/'
    filename_vh = './data/test_features/'
    jrc_change = './data/jrc_gsw_change/'
    jrc_transitions = './data/jrc_transitions/'
    jrc_seasonality = './data/jrc_seasonality/'
    jrc_recurrence = './data/jrc_recurrence/'
    jrc_occurrence = './data/jrc_occurrence/'
    jrc_extent = './data/jrc_extent/'
    nasadem = './data/nasadem/'

    visual_reslut = './data/'

    if not os.path.exists(visual_reslut):
        os.mkdir(visual_reslut)

    for img_name in tqdm(os.listdir(nasadem)):

        with rasterio.open(os.path.join(filename_vv, img_name.replace('.tif', '_vh.tif'))) as vv:
            vv_path = vv.read(1)
        with rasterio.open(os.path.join(filename_vh, img_name.replace('.tif', '_vv.tif'))) as vh:
            vh_path = vh.read(1)
        with rasterio.open(os.path.join(jrc_change, img_name)) as gc:
            jrc_gsw_change = gc.read(1)
        with rasterio.open(os.path.join(jrc_transitions, img_name)) as gt:
            jrc_gsw_transitions = gt.read(1)
        with rasterio.open(os.path.join(jrc_seasonality, img_name)) as gs:
            jrc_gsw_seasonality = gs.read(1)
        with rasterio.open(os.path.join(jrc_recurrence, img_name)) as gr:
            jrc_gsw_recurrence = gr.read(1)
        with rasterio.open(os.path.join(jrc_occurrence, img_name)) as go:
            jrc_gsw_occurrence = go.read(1)
        with rasterio.open(os.path.join(jrc_extent, img_name)) as ge:
            jrc_gsw_extent = ge.read(1)
        with rasterio.open(os.path.join(nasadem, img_name)) as nsd:
            nasadem = nsd.read(1)

        img = np.stack([vv_path, vh_path], axis=-1)
        img_auxiliary = np.stack([jrc_gsw_change, jrc_gsw_transitions, jrc_gsw_seasonality, jrc_gsw_recurrence,
                                  jrc_gsw_occurrence, jrc_gsw_extent, nasadem], axis=-1)

        # Min-max normalization
        min_aux = np.min(img_auxiliary)
        max_aux = np.max(img_auxiliary)
        img_auxiliary = (img_auxiliary - min_aux) / (max_aux - min_aux)

        # Min-max normalization
        min_norm = -77
        max_norm = 26
        img = np.clip(img, min_norm, max_norm)
        img = (img - min_norm) / (max_norm - min_norm)

        img = np.concatenate((img, img_auxiliary), axis=-1)

        result = inference_segmentor(segmentor, img)
        res_img = visual_result(segmentor, result, palette=[[0,0, 0], [225, 225, 225]])
        cv2.imwrite(os.path.join(visual_reslut, img_name), res_img)
