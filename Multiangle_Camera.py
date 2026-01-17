"""
SynvowMultiangle Node for ComfyUI
A 3D camera control node that outputs angle prompts
"""

import os
import time
import numpy as np
from PIL import Image
import torch
import folder_paths


class SynvowMultiangleCameraNode:
    """
    3D Camera Angle Control Node
    Provides a 3D scene to adjust camera angles and outputs a formatted prompt string
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "horizontal_angle": ("INT", {
                    "default": 0, "min": 0, "max": 360, "step": 1, "display": "slider"
                }),
                "vertical_angle": ("INT", {
                    "default": 0, "min": -60, "max": 60, "step": 1, "display": "slider"
                }),
                "zoom": ("FLOAT", {
                    "default": 5.0, "min": 0.0, "max": 10.0, "step": 0.1, "display": "slider"
                }),
                "size_mode": (["custom", "original"], {"default": "custom"}),
                "max_long_edge": ("INT", {
                    "default": 1024, "min": 64, "max": 8192, "step": 8
                }),
            },
            "optional": {
                "image": ("IMAGE",),
            },
            "hidden": {
                "uploaded_image": "STRING",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("prompt", "image")
    FUNCTION = "generate_prompt"
    CATEGORY = "💫SynVow/Image"
    DESCRIPTION = "3D camera angle control node for multi-angle prompt generation"

    def generate_prompt(self, horizontal_angle, vertical_angle, zoom,
                        size_mode="custom", max_long_edge=1024,
                        image=None, uploaded_image="", prompt=None, extra_pnginfo=None):
        # Validate input ranges
        horizontal_angle = max(0, min(360, int(horizontal_angle)))
        vertical_angle = max(-60, min(60, int(vertical_angle)))
        zoom = max(0.0, min(10.0, float(zoom)))

        h_angle = horizontal_angle % 360

        # Azimuth (horizontal) - 8 directions
        h_boundaries = [22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5]
        h_directions = [
            "front view", "front-right quarter view", "right side view",
            "back-right quarter view", "back view", "back-left quarter view",
            "left side view", "front-left quarter view"
        ]
        h_idx = 0
        for i, b in enumerate(h_boundaries):
            if h_angle < b:
                h_idx = i
                break
        else:
            h_idx = 0
        h_direction = h_directions[h_idx]

        # Elevation (vertical) - 5 levels
        if vertical_angle < -30:
            v_direction = "extreme low-angle shot"
        elif vertical_angle < -10:
            v_direction = "low-angle shot"
        elif vertical_angle < 10:
            v_direction = "eye-level shot"
        elif vertical_angle < 40:
            v_direction = "elevated shot"
        else:
            v_direction = "high-angle shot"

        # Distance - 3 levels
        if zoom < 2:
            distance = "wide shot"
        elif zoom < 6:
            distance = "medium shot"
        else:
            distance = "close-up"

        prompt = f"<sks> {h_direction} {v_direction} {distance}"

        # Get image: prioritize connected IMAGE input, fallback to uploaded file
        if image is not None:
            img_tensor = image[0] if len(image.shape) == 4 else image
        elif uploaded_image:
            image_path = folder_paths.get_annotated_filepath(uploaded_image)
            pil_img = Image.open(image_path).convert('RGB')
            img_np = np.array(pil_img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np)
        else:
            raise ValueError("Please provide an image: connect image input or use the load button")

        # Process image based on size mode
        if size_mode == "custom":
            h, w = img_tensor.shape[:2]
            long_edge = max(h, w)
            if long_edge != max_long_edge:
                scale = max_long_edge / long_edge
                new_h, new_w = int(h * scale), int(w * scale)
                img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
                pil_image = Image.fromarray(img_np).resize((new_w, new_h), Image.BILINEAR)
                img_np = np.array(pil_image).astype(np.float32) / 255.0
                output_image = torch.from_numpy(img_np).unsqueeze(0)
            else:
                output_image = img_tensor.unsqueeze(0)
        else:
            output_image = img_tensor.unsqueeze(0)

        # Save preview image to temp directory for frontend 3D preview
        preview_image = None
        try:
            temp_dir = folder_paths.get_temp_directory()
            timestamp = int(time.time() * 1000)
            filename = f"synvow_preview_{timestamp}.png"
            filepath = os.path.join(temp_dir, filename)

            img_np = (output_image[0].cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            pil_img.save(filepath, compress_level=1)

            preview_image = {"filename": filename, "subfolder": "", "type": "temp"}
        except Exception as e:
            print(f"[SynvowMultiangle] Error saving preview: {e}")

        return {
            "ui": {"preview_image": [preview_image] if preview_image else []},
            "result": (prompt, output_image)
        }


NODE_CLASS_MAPPINGS = {
    "SynvowMultiangleCameraNode": SynvowMultiangleCameraNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SynvowMultiangleCameraNode": "Synvow Multiangle Camera",
}

