"""Libero policy with next action prediction support."""

import dataclasses
from typing import Any

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    """Parse image to ensure correct format."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class LiberoNextActionInputs(transforms.DataTransformFn):
    """Transform Libero data for next action prediction (o_t, a_t -> a_{t+1})."""

    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0_FAST

    def __call__(self, data: dict) -> dict:
        """Transform data for next action prediction.

        This transform:
        1. Keeps o_t (current observation) as is
        2. Stores a_t[horizon] as current_actions (input)
        3. Gets a_{t+1}[horizon] as target (next action horizon)
        """
        mask_padding = self.model_type == _model.ModelType.PI0

        # Process state
        state = transforms.pad_to_dim(data.get("state", data.get("observation/state", np.zeros(8))), self.action_dim)

        # Process images
        base_image = _parse_image(data.get("image", data.get("observation/image", np.zeros((224, 224, 3)))))
        wrist_image = _parse_image(
            data.get("wrist_image", data.get("observation/wrist_image", np.zeros((224, 224, 3))))
        )

        # Create base inputs
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_ if mask_padding else np.True_,
            },
        }

        # Handle actions for next action prediction
        # The NextActionTransform should have already created current_actions and next actions
        if "current_actions" in data:
            # Current action horizon as input
            inputs["current_actions"] = transforms.pad_to_dim(data["current_actions"], self.action_dim)
        elif "actions" in data:
            # Fallback if NextActionTransform wasn't applied
            actions = np.asarray(data["actions"])
            inputs["current_actions"] = transforms.pad_to_dim(actions, self.action_dim)

        if "actions" in data:
            # Target actions (should be next action horizon from NextActionTransform)
            inputs["actions"] = transforms.pad_to_dim(data["actions"], self.action_dim)
            
        if "next_action_mask" in data:
            inputs["next_action_mask"] = data["next_action_mask"]

        # Pass prompt if available
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoNextActionOutputs(transforms.DataTransformFn):
    """Output transform for next action prediction."""

    action_dim: int = 7

    def __call__(self, data: dict) -> dict:
        """Extract next actions from model output."""
        # Return only the relevant action dimensions (first 7 for Libero)
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
