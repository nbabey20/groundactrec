# Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""Demo CoMotion with a video file or a directory of images."""

import logging
import os
import shutil
import tempfile
from pathlib import Path

import click
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from comotion_demo.models import comotion
from comotion_demo.utils import dataloading, helper
from comotion_demo.utils import track as track_utils

try:
    from aitviewer.configuration import CONFIG
    from aitviewer.headless import HeadlessRenderer
    from aitviewer.renderables.billboard import Billboard
    from aitviewer.renderables.smpl import SMPLLayer, SMPLSequence
    from aitviewer.scene.camera import OpenCVCamera

    comotion_model_dir = Path(comotion.__file__).parent
    CONFIG.smplx_models = os.path.join(comotion_model_dir, "../data")
    CONFIG.window_type = "pyqt6"
    aitviewer_available = True
except ModuleNotFoundError:
    print("WARNING: Skipped aitviewer import, ensure it is installed to run visualization.")
    aitviewer_available = False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# -----------------------------------------------------------------------------
# Runtime flags (may be toggled later)
# -----------------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_fp16 = False  # set True later if you want FP16
use_mps = torch.mps.is_available()

# Ensure normalization tensors live on the same device as the frames
# Make normalization device-agnostic
def _normalize(img: torch.Tensor) -> torch.Tensor:
    return (img - dataloading.IMG_MEAN.to(img.device, dtype=img.dtype)) / \
           dataloading.IMG_STD.to(img.device, dtype=img.dtype)

dataloading.normalize_image = _normalize  # override original fp32 version

# Ensure dataloading recognizes common video extensions (incl. .avi)
from comotion_demo.utils import dataloading as _dl
if hasattr(_dl, "VIDEO_EXTENSIONS"):
    try:
        exts = set(x.lower() for x in _dl.VIDEO_EXTENSIONS)
    except TypeError:
        exts = set(map(str.lower, list(_dl.VIDEO_EXTENSIONS)))
    exts.update({".avi", ".mp4", ".mov", ".mkv"})
    _dl.VIDEO_EXTENSIONS = tuple(sorted(exts))



def prepare_scene(viewer, width, height, K, image_paths, fps=30):
    """Prepare the scene for AITViewer rendering."""
    viewer.reset()
    viewer.scene.floor.enabled = False
    viewer.scene.origin.enabled = False
    extrinsics = np.eye(4)[:3]

    # Initialize camera
    cam = OpenCVCamera(K, extrinsics, cols=width, rows=height, viewer=viewer)
    viewer.scene.add(cam)
    viewer.scene.camera.position = [0, 0, -5]
    viewer.scene.camera.target = [0, 0, 10]
    viewer.auto_set_camera_target = False
    viewer.set_temp_camera(cam)
    viewer.playback_fps = fps

    # "billboard" display for video frames
    billboard = Billboard.from_camera_and_distance(
        cam, 100.0, cols=width, rows=height, textures=image_paths
    )
    viewer.scene.add(billboard)


def add_pose_to_scene(
    viewer,
    smpl_layer,
    betas,
    pose,
    trans,
    color=(0.6, 0.6, 0.6),
    alpha=1,
    color_ref=None,
):
    """Add estimated poses to the rendered scene."""
    if betas.ndim == 2:
        betas = betas[None]
        pose = pose[None]
        trans = trans[None]

    poses_root = pose[..., :3]
    poses_body = pose[..., 3:]
    max_people = pose.shape[1]

    if (betas != 0).any():
        for person_idx in range(max_people):
            if color_ref is None:
                person_color = color
            else:
                person_color = color_ref[person_idx % len(color_ref)] * 0.4 + 0.3
            person_color = [c_ for c_ in person_color] + [alpha]

            valid_vals = (betas[:, person_idx] != 0).any(-1)
            idx_range = valid_vals.nonzero()
            if len(idx_range) > 0:
                trans[~valid_vals][..., 2] = -10000
                viewer.scene.add(
                    SMPLSequence(
                        smpl_layer=smpl_layer,
                        betas=betas[:, person_idx],
                        poses_root=poses_root[:, person_idx],
                        poses_body=poses_body[:, person_idx],
                        trans=trans[:, person_idx],
                        color=person_color,
                    )
                )


def visualize_poses(
    input_path,
    cache_path,
    video_path,
    start_frame,
    num_frames,
    frameskip=1,
    color=(0.6, 0.6, 0.6),
    alpha=1,
    fps=30,
):
    """Visualize SMPL poses."""
    logging.info(f"Rendering SMPL video: {input_path}")

    # Prepare temporary directory with saved images
    tmp_vis_dir = Path(tempfile.mkdtemp())

    frame_idx = 0
    image_paths = []
    for image, K in dataloading.yield_image_and_K(
        input_path, start_frame, num_frames, frameskip
    ):
        image_height, image_width = image.shape[-2:]
        image = dataloading.convert_tensor_to_image(image)
        image_paths.append(f"{tmp_vis_dir}/{frame_idx:06d}.jpg")
        Image.fromarray(image).save(image_paths[-1])
        frame_idx += 1

    # Initialize viewer
    viewer = HeadlessRenderer(size=(image_width, image_height))

    if dataloading.is_a_video(input_path):
        fps = int(dataloading.get_input_video_fps(input_path))

    prepare_scene(viewer, image_width, image_height, K.cpu().numpy(), image_paths, fps)

    # Prepare SMPL poses
    smpl_layer = SMPLLayer(model_type="smpl", gender="neutral")
    if not cache_path.exists():
        logging.warning("No detections found.")
    else:
        preds = torch.load(cache_path, weights_only=False, map_location="cpu")
        track_subset = track_utils.query_range(preds, 0, frame_idx - 1)
        id_lookup = track_subset["id"].max(0)[0]
        color_ref = helper.color_ref[id_lookup % len(helper.color_ref)]
        if len(id_lookup) == 1:
            color_ref = [color_ref]

        betas = track_subset["betas"]
        pose = track_subset["pose"]
        trans = track_subset["trans"]

        add_pose_to_scene(
            viewer, smpl_layer, betas, pose, trans, color, alpha, color_ref
        )

    # Save rendered scene
    viewer.save_video(
        video_dir=str(video_path),
        output_fps=fps,
        ensure_no_overwrite=False,
    )

    # Remove temporary directory
    shutil.rmtree(tmp_vis_dir)


def run_detection(input_path, cache_path, skip_visualization=False, model=None):
    """Run model and visualize detections on single image."""
    if model is None:
        model = comotion.CoMotion(use_coreml=use_mps)
    model.to(device).eval()

    # Load image
    image = np.array(Image.open(input_path))
    image = dataloading.convert_image_to_tensor(image)
    K = dataloading.get_default_K(image)
    cropped_image, cropped_K = dataloading.prepare_network_inputs(image, K, device)

    # Get detections
    detections = model.detection_model(cropped_image, cropped_K)
    detections = comotion.detect.decode_network_outputs(
        K.to(device),
        model.smpl_decoder,
        detections,
        std=0.15,   # Adjust NMS sensitivity
        conf_thr=0.25,  # Adjust confidence cutoff
    )

    detections = {k: v[0].cpu() for k, v in detections.items()}
    torch.save(detections, cache_path)

    if not skip_visualization:
        # Initialize viewer
        image_height, image_width = image.shape[-2:]
        viewer = HeadlessRenderer(size=(image_width, image_height))
        prepare_scene(
            viewer, image_width, image_height, K.cpu().numpy(), [str(input_path)]
        )

        # Prepare SMPL poses
        smpl_layer = SMPLLayer(model_type="smpl", gender="neutral")
        add_pose_to_scene(
            viewer,
            smpl_layer,
            detections["betas"],
            detections["pose"],
            detections["trans"],
        )

        # Save rendered scene
        viewer.save_frame(str(cache_path).replace(".pt", ".png"))


def track_poses(input_path, cache_path, start_frame, num_frames, frameskip=1, model=None):
    """Track poses over a video or a directory of images."""
    if model is None:
        model = comotion.CoMotion(use_coreml=use_mps)

    # ── 1. put the model on GPU and register the device inside it ────────────
    model = model.to(device).eval()
    model.device = device  # important for internal calls
    if use_fp16:
        model.half()

    detections, tracks = [], []
    initialized = False
    for image, K in tqdm(
        dataloading.yield_image_and_K(input_path, start_frame, num_frames, frameskip),
        desc="Running CoMotion",
    ):
        # move to device
        image, K = image.to(device), K.to(device)

        if use_fp16:
            image, K = image.half(), K.half()

        if not initialized:
            model.init_tracks(image.shape[-2:])
            initialized = True

        if use_fp16:
            with torch.cuda.amp.autocast():
                detection, track = model(image, K, use_mps=use_mps)
        else:
            detection, track = model(image, K, use_mps=use_mps)

        detections.append({k: v.cpu() for k, v in detection.items()})
        tracks.append(track.cpu())

    # -------------- post-processing (unchanged) -----------------------------
    detections = {k: [d[k] for d in detections] for k in detections[0]}
    tracks_t = torch.stack(tracks, 1)
    tracks = {k: getattr(tracks_t, k) for k in ["id", "pose", "trans", "betas"]}

    K_cpu = K.cpu()
    image_res = image.shape[-2:]
    track_ref = track_utils.cleanup_tracks(
        {"detections": detections, "tracks": tracks},
        K_cpu,
        model.smpl_decoder.cpu(),
        min_matched_frames=1,
    )
    if track_ref:
        f_idx, t_idx = track_utils.convert_to_idxs(
            track_ref, tracks["id"][0].squeeze(-1).long()
        )
        preds = {k: v[0, f_idx, t_idx] for k, v in tracks.items()}
        preds["id"] = preds["id"].squeeze(-1).long()
        preds["frame_idx"] = f_idx
        torch.save(preds, cache_path)

        bboxes = track_utils.bboxes_from_smpl(
            model.smpl_decoder,
            {k: preds[k] for k in ["betas", "pose", "trans"]},
            image_res,
            K_cpu,
        )
        with open(str(cache_path).replace(".pt", ".txt"), "w") as f:
            f.write(track_utils.convert_to_mot(preds["id"], preds["frame_idx"], bboxes))


@click.command()
@click.option(
    "-i",
    "--input-path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to the input video, a directory of images, or a single input image.",
)
@click.option(
    "-o",
    "--output-dir",
    required=True,
    type=click.Path(exists=False, path_type=Path),
    help="Path to the output directory.",
)
@click.option(
    "-s",
    "--start-frame",
    default=0,
    type=int,
    help="Frame to start with.",
)
@click.option(
    "-n",
    "--num-frames",
    default=1_000_000_000,
    type=int,
    help="Number of frames to process.",
)
@click.option(
    "--skip-visualization",
    is_flag=True,
    help="Whether to skip rendering the output SMPL meshes.",
)
@click.option(
    "--frameskip",
    default=1,
    type=int,
    help="Subsample video frames (e.g. frameskip=2 processes every other frame).",
)
def main(input_path, output_dir, start_frame, num_frames, skip_visualization, frameskip):
    """Demo entry point."""
    output_dir.mkdir(parents=True, exist_ok=True)
    input_name = input_path.stem
    skip_visualization = skip_visualization | (not aitviewer_available)

    cache_path = output_dir / f"{input_name}.pt"
    if input_path.suffix.lower() in dataloading.IMAGE_EXTENSIONS:
        # Run and visualize detections for a single image
        run_detection(input_path, cache_path, skip_visualization)
    else:
        # Run unrolled tracking on a full video
        track_poses(input_path, cache_path, start_frame, num_frames, frameskip)
        if not skip_visualization:
            video_path = output_dir / f"{input_name}.mp4"
            visualize_poses(
                input_path, cache_path, video_path, start_frame, num_frames, frameskip
            )


if __name__ == "__main__":
    main()
