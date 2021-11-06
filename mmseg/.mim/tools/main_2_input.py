import os
import mmcv
import torch
import cv2
import os.path as osp
import numpy as np
from inference import init_segmentor, visual_result, inference_segmentor
import time
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from tifffile import imwrite
import typer

from flood_model import FloodModel


os.environ["CUDA_VISIBLE_DEVICES"] = "0"



ROOT_DIRECTORY = Path("/codeexecution")
SUBMISSION_DIRECTORY = ROOT_DIRECTORY / "submission"
ASSETS_DIRECTORY = ROOT_DIRECTORY / "assets"
DATA_DIRECTORY = ROOT_DIRECTORY / "data"

input_images = DATA_DIRECTORY / "test_features"
jrc_change = DATA_DIRECTORY / 'jrc_gsw_change'
jrc_transitions = DATA_DIRECTORY / 'jrc_transitions'
jrc_seasonality = DATA_DIRECTORY / 'jrc_seasonality'
jrc_recurrence = DATA_DIRECTORY / 'jrc_recurrence'
jrc_occurrence = DATA_DIRECTORY / 'jrc_occurrence'
jrc_extent = DATA_DIRECTORY / 'jrc_extent'
nasadem = DATA_DIRECTORY / 'nasadem'


# make sure the smp loader can find our torch assets because we don't have internet!
os.environ["TORCH_HOME"] = str(ASSETS_DIRECTORY / "torch")


def make_prediction(chip_id, segmentor):
    """
    Given a chip_id, read in the vv/vh bands and predict a water mask.
    Args:
        chip_id (str): test chip id
    Returns:
        output_prediction (arr): prediction as a numpy array
    """
    logger.info("Starting inference.")
    try:
        with rasterio.open(os.path.join(filename_vv, chip_id.replace('.tif', '_vh.tif'))) as vv:
            vv_path = vv.read(1)
        with rasterio.open(os.path.join(filename_vh, chip_id.replace('.tif', '_vv.tif'))) as vh:
            vh_path = vh.read(1)

        img = np.stack([vv_path, vh_path], axis=-1)

        # Min-max normalization
        min_norm = -77
        max_norm = 26
        img = np.clip(img, min_norm, max_norm)
        img = (img - min_norm) / (max_norm - min_norm)

        result = inference_segmentor(segmentor, img)
        seg = result[0]

    except Exception as e:
        logger.error(f"No bands found for {chip_id}. {e}")
        raise
    return seg


def get_expected_chip_ids():
    """
    Use the test features directory to see which images are expected.
    """
    paths = INPUT_IMAGES_DIRECTORY.glob("*.tif")
    # Return one chip id per two bands (VV/VH)
    ids = list(sorted(set(path.stem.split("_")[0] for path in paths)))
    return ids


def main():
    """
    For each set of two input bands, generate an output file
    using the `make_predictions` function.
    """
    logger.info("Loading model")
    # explicitly set where we expect smp to load the saved resnet from just to be sure
    torch.hub.set_dir(ASSETS_DIRECTORY / "torch/hub")

    config_file = "./assets/model_weight/deeplabv3_r50-d8_512x512_80k_flood_data_test_2.py"
    checkpoint_file = "./assets/model_weight/iter_40000_2.pth"   # checkpoint_file = '/data/projects/mmdetection-1.1.0/work_dirs/bankcard_6_11/epoch_40.pth'

    segmentor = init_segmentor(config_file, checkpoint_file)

    logger.info("Finding chip IDs")
    chip_ids = get_expected_chip_ids()
    if not chip_ids:
        typer.echo("No input images found!")
        raise typer.Exit(code=1)

    logger.info(f"Found {len(chip_ids)} test chip_ids. Generating predictions.")
    for chip_id in tqdm(chip_ids, miniters=25):
        output_path = SUBMISSION_DIRECTORY / f"{chip_id}.tif"
        output_data = make_prediction(chip_id, segmentor).astype(np.uint8)
        imwrite(output_path, output_data, dtype=np.uint8)

    logger.success(f"Inference complete.")


if __name__ == "__main__":
    typer.run(main)



