import json
from pathlib import Path
import torch

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# torch.multiprocessing.set_start_method("spawn")

import colorsys
import datetime
import os
import subprocess

import cv2
import gradio as gr
import imageio.v2 as iio
import numpy as np

from loguru import logger as guru

from sam2.build_sam import build_sam2_video_predictor


class PromptGUI(object):
    def __init__(self, checkpoint_dir, model_cfg):
        self.checkpoint_dir = checkpoint_dir
        self.model_cfg = model_cfg
        self.sam_model = None
        self.tracker = None

        self.selected_points = []
        self.selected_labels = []
        self.cur_label_val = 1.0

        self.frame_index = 0
        self.image = None
        self.cur_mask_idx = 0
        # can store multiple object masks
        # saves the masks and logits for each mask index
        self.cur_masks = {}
        self.cur_logits = {}
        self.index_masks_all = []
        self.color_masks_all = []

        self.img_dir = ""
        self.img_paths = []
        self.init_sam_model()

    def init_sam_model(self):
        if self.sam_model is None:
            self.sam_model = build_sam2_video_predictor(self.model_cfg, self.checkpoint_dir)
            guru.info(f"loaded model checkpoint {self.checkpoint_dir}")

    def clear_points(self) -> tuple[None, None, str]:
        self.selected_points.clear()
        self.selected_labels.clear()
        message = "Cleared points, select new points to update mask"
        return self.image, None, message

    def add_new_mask(self):
        self.cur_mask_idx += 1
        self.clear_points()
        message = f"Creating new mask with index {self.cur_mask_idx}"
        return None, message

    def make_index_mask(self, masks):
        assert len(masks) > 0
        idcs = list(masks.keys())
        idx_mask = masks[idcs[0]].astype("uint8")
        for i in idcs:
            mask = masks[i]
            idx_mask[mask] = i + 1
        return idx_mask

    def _clear_image(self):
        """
        clears image and all masks/logits for that image
        """
        self.image = None
        self.cur_mask_idx = 0
        self.frame_index = 0
        self.cur_masks = {}
        self.cur_logits = {}
        self.index_masks_all = []
        self.color_masks_all = []

    def reset(self):
        self._clear_image()
        self.sam_model.reset_state(self.inference_state)

    def set_img_dir(self, img_dir: str) -> int:
        self._clear_image()
        self.img_dir = img_dir
        self.img_paths = [
            f"{img_dir}/{p}" for p in sorted(os.listdir(img_dir)) if isimage(p)
        ]
        guru.debug(f"loaded {len(self.img_paths)} in image dir {img_dir}")

        return len(self.img_paths)

    def set_input_image(self, i: int = 0) -> np.ndarray | None:
        guru.debug(f"Setting frame {i} / {len(self.img_paths)}")
        if i < 0 or i >= len(self.img_paths):
            return self.image
        self.clear_points()
        self.frame_index = i
        image = iio.imread(self.img_paths[i])
        self.image = image

        return image

    def get_sam_features(self) -> tuple[str, np.ndarray | None]:
        guru.debug(f"Getting sam features")
        self.inference_state = self.sam_model.init_state(video_path=self.img_dir)
        # self.sam_model.reset_state(self.inference_state)
        msg = (
            "SAM features extracted. "
            "Click input image to add points, update mask, and submit when ready to start tracking"
        )
        guru.debug(f"image shape: {self.image.shape}")
        return msg

    def set_positive(self) -> str:
        self.cur_label_val = 1.0
        guru.debug("Selected positive label")
        return "Selecting positive points. Submit the mask to start tracking"

    def set_negative(self) -> str:
        self.cur_label_val = 0.0
        guru.debug("Selected negative label")
        return "Selecting negative points. Submit the mask to start tracking"

    def add_point(self, frame_idx, i, j):
        """
        get the index mask of the objects
        """
        self.selected_points.append([j, i])
        self.selected_labels.append(self.cur_label_val)
        # masks, scores, logits if we want to update the mask
        masks = self.get_sam_mask(
            frame_idx, np.array(self.selected_points, dtype=np.float32), np.array(self.selected_labels, dtype=np.int32)
        )
        mask = self.make_index_mask(masks)

        return mask

    def get_sam_mask(self, frame_idx, input_points, input_labels):
        """
        :param frame_idx int
        :param input_points (np array) (N, 2)
        :param input_labels (np array) (N,)
        return (H, W) mask, (H, W) logits
        """
        assert self.sam_model is not None

        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, out_obj_ids, out_mask_logits = self.sam_model.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=frame_idx,
                obj_id=self.cur_mask_idx,
                points=input_points,
                labels=input_labels,
            )

        return {
            out_obj_id: (out_mask_logits[i] > 0.0).squeeze().cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    def run_tracker(self) -> tuple[str, str]:

        # read images and drop the alpha channel
        images = [iio.imread(p)[:, :, :3] for p in self.img_paths]

        video_segments = {}  # video_segments contains the per-frame segmentation results
        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for out_frame_idx, out_obj_ids, out_mask_logits in self.sam_model.propagate_in_video(self.inference_state,
                                                                                                 start_frame_idx=0):
                masks = {
                    out_obj_id: (out_mask_logits[i] > 0.0).squeeze().cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
                video_segments[out_frame_idx] = masks
            # index_masks_all.append(self.make_index_mask(masks))

        self.index_masks_all = [self.make_index_mask(v) for k, v in video_segments.items()]

        out_frames, self.color_masks_all = colorize_masks(images, self.index_masks_all)
        out_vidpath = "tracked_colors.mp4"
        iio.mimwrite(out_vidpath, out_frames)
        message = f"Wrote current tracked video to {out_vidpath}."
        instruct = "Save the masks to an output directory if it looks good!"
        return out_vidpath, f"{message} {instruct}"

    def save_masks_to_dir(self, output_dir: str) -> str:
        assert self.color_masks_all is not None
        meta_file = os.path.join(output_dir, "../space.json")
        have_meta = False
        if os.path.exists(meta_file) and os.path.isfile(meta_file):
            have_meta = True
            with open(meta_file, "r") as f:
                space = json.load(f)
            frames = sorted(space["frames"], key=lambda x: x["file_path"])
        os.makedirs(output_dir, exist_ok=True)
        idx = 0
        for img_path, clr_mask, id_mask in zip(self.img_paths, self.color_masks_all, self.index_masks_all):
            name = os.path.basename(img_path)
            out_path = f"{output_dir}/{name}"
            iio.imwrite(out_path, clr_mask)
            np_out_path = f"{output_dir}/{name[:-4]}.npy"
            if have_meta:
                frames[idx].update(
                    {"mask_file_path": os.path.join(Path(output_dir).name, Path(np_out_path).name)})
            np.save(np_out_path, id_mask)
            idx += 1

        if have_meta:
            space["frames"] = frames
            with open(meta_file, "w") as f:
                json.dump(space, f)

        message = f"Saved masks to {output_dir}!"
        guru.debug(message)
        return message


def isimage(p):
    ext = os.path.splitext(p.lower())[-1]
    return ext in [".png", ".jpg", ".jpeg"]


def draw_points(img, points, labels):
    out = img.copy()
    for p, label in zip(points, labels):
        x, y = int(p[0]), int(p[1])
        color = (0, 255, 0) if label == 1.0 else (255, 0, 0)
        out = cv2.circle(out, (x, y), 10, color, -1)
    return out


def get_hls_palette(
        n_colors: int,
        lightness: float = 0.5,
        saturation: float = 0.7,
) -> np.ndarray:
    """
    returns (n_colors, 3) tensor of colors,
        first is black and the rest are evenly spaced in HLS space
    """
    hues = np.linspace(0, 1, int(n_colors) + 1)[1:-1]  # (n_colors - 1)
    # hues = (hues + first_hue) % 1
    palette = [(0.0, 0.0, 0.0)] + [
        colorsys.hls_to_rgb(h_i, lightness, saturation) for h_i in hues
    ]
    return (255 * np.asarray(palette)).astype("uint8")


def colorize_masks(images, index_masks, fac: float = 0.5):
    max_idx = max([m.max() for m in index_masks])
    guru.debug(f"{max_idx=}")
    palette = get_hls_palette(max_idx + 1)
    color_masks = []
    out_frames = []
    for img, mask in zip(images, index_masks):
        clr_mask = palette[mask.astype("int")]
        color_masks.append(clr_mask)
        out_u = compose_img_mask(img, clr_mask, fac)
        out_frames.append(out_u)
    return out_frames, color_masks


def compose_img_mask(img, color_mask, fac: float = 0.5):
    out_f = fac * img / 255 + (1 - fac) * color_mask / 255
    out_u = (255 * out_f).astype("uint8")
    return out_u


def listdir(vid_dir):
    if vid_dir is not None and os.path.isdir(vid_dir):
        return sorted(os.listdir(vid_dir))
    return []


def make_demo(checkpoint_dir, model_cfg):
    prompts = PromptGUI(checkpoint_dir, model_cfg)
    start_instructions = (
        "Select a video file to extract frames from, "
        "or select an image directory with frames already extracted."
    )

    with gr.Blocks() as demo:
        instruction = gr.Textbox(
            start_instructions, label="Instruction", interactive=False
        )
        with gr.Tab("Input images"):
            with gr.Row():
                with gr.Column():
                    image_dir_textbox = gr.Textbox(value=None, label="Input image directory", interactive=True)

                    # Initialize slider range to update dynamically
                    frame_index_slider = gr.Slider(
                        label="Frame index",
                        minimum=0,
                        maximum=len(prompts.img_paths) - 1,
                        value=0,
                        step=1,
                    )

                    with gr.Row():
                        point_type = gr.Radio(label="point type", choices=["include", "exclude"], value="include")
                        clear_points_btn = gr.Button("Clear Points")
                    # checkpoint = gr.Dropdown(label="Checkpoint", choices=["tiny", "small", "base-plus", "large"],
                    #                          value="tiny")
                    submit_button = gr.Button("Submit mask for tracking")
                    with gr.Row():
                        mask_dir_field = gr.Text(
                            value="", label="Path to save masks", interactive=True
                        )
                        save_button = gr.Button("Save masks")

                with gr.Column():
                    input_image = gr.Image(
                        prompts.set_input_image(0),  # Static, needs to be dynamic
                        label="Input Frame",
                        every=1,
                    )

                    final_video = gr.Video(label="Masked video")

        with gr.Tab("Input video"):
            video_file_field = gr.Video(label="Upload video")

        # Function to dynamically update slider and image
        def update_image_dir(img_dir: str):
            num_img = prompts.set_img_dir(img_dir)
            slider_update = gr.update(maximum=num_img - 1, value=0)
            image_update = gr.update(value=prompts.set_input_image(0))  # Reset image to first frame
            mask_dir_update = gr.update(value=str(Path(img_dir).parent / "mask"))
            msg = (
                f"Loaded {num_img} images from {img_dir}. Choose a frame to run SAM!"
            )
            return slider_update, image_update, mask_dir_update, msg

        # Function to dynamically update image based on slider value
        def update_image(frame_index):
            return gr.update(value=prompts.set_input_image(frame_index))

        def update_point_type(point_type: str):
            assert point_type in ["include", "exclude"]
            if point_type == "include":
                prompts.set_positive()
            else:
                prompts.set_negative()
            return

        def get_select_coords(frame_idx, img, evt: gr.SelectData):
            i = evt.index[1]  # type: ignore
            j = evt.index[0]  # type: ignore
            index_mask = prompts.add_point(frame_idx, i, j)
            guru.debug(f"{index_mask.shape=}")
            palette = get_hls_palette(index_mask.max() + 1)
            color_mask = palette[index_mask]
            out_u = compose_img_mask(img, color_mask)
            out = draw_points(out_u, prompts.selected_points, prompts.selected_labels)
            return out

        # Attach event handlers
        image_dir_textbox.submit(
            update_image_dir,
            [image_dir_textbox],
            [frame_index_slider, input_image, mask_dir_field, instruction]  # Include input_image to reset it
        ).then(prompts.get_sam_features, outputs=[instruction])

        frame_index_slider.change(
            prompts.set_input_image,
            [frame_index_slider],
            [input_image]
        )

        input_image.select(get_select_coords,
                    [frame_index_slider, input_image],
                    [input_image])

        point_type.change(
            update_point_type,
            [point_type],
        )

        clear_points_btn.click(prompts.clear_points,
                               outputs=[input_image, final_video, instruction])

        submit_button.click(prompts.run_tracker, outputs=[final_video, instruction])
        save_button.click(
            prompts.save_masks_to_dir, [mask_dir_field], outputs=[instruction]
        )

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/sam2_hiera_tiny.pt")
    parser.add_argument("--model_cfg", type=str, default="sam2_hiera_t.yaml")
    args = parser.parse_args()

    # device = "cuda" if torch.cuda.is_available() else "cpu"

    demo = make_demo(
        args.checkpoint_dir,
        args.model_cfg,
    )
    demo.launch(server_port=args.port)
