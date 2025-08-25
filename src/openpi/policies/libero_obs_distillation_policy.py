"""Libero policy for observation distillation (o_{t-1}, a_{t-1} → a_t)."""

import dataclasses
from typing import Any, Dict

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
class ObservationDistillationTransform(transforms.DataTransformFn):
    """Transform that shifts observations and actions for distillation.

    This creates pairs of:
    - Previous: (o_{t-1}, a_{t-1})
    - Current: o_t
    - Target: a_t

    For training observation distillation where the student uses previous
    observation and action to predict current action, while teacher uses
    current observation.
    """

    action_dim: int
    handle_first_timestep: str = "zero"  # Options: "duplicate", "zero", "skip"

    def __call__(self, data: dict) -> dict:
        """Apply temporal shifting for observation distillation.

        Args:
            data: Input data with observations and actions at time t

        Returns:
            Modified data with:
            - prev_observation: o_{t-1}
            - prev_actions: a_{t-1}
            - observation: o_t (for teacher)
            - actions: a_t (target)
        """
        result = data.copy()

        # Handle trajectory data if available
        if "trajectory_data" in data and data.get("timestep", 0) > 0:
            # We have access to previous timestep
            trajectory = data["trajectory_data"]
            timestep = data["timestep"]

            # Get previous observation and action
            result["prev_observation"] = trajectory["observations"][timestep - 1]
            result["prev_actions"] = trajectory["actions"][timestep - 1]

            # Current observation stays as is
            # Current action is the target

        else:
            # Need to handle first timestep or when no trajectory data
            if self.handle_first_timestep == "duplicate":
                # Use current observation as previous (teacher and student see same)
                result["prev_observation"] = data.get("observation", data)
                # Zero previous action, preserving action horizon shape if present
                if "actions" in data and len(data["actions"].shape) == 2:
                    # Actions have horizon dimension [horizon, action_dim]
                    result["prev_actions"] = np.zeros_like(data["actions"])
                else:
                    # Single timestep action
                    result["prev_actions"] = np.zeros(self.action_dim)

            elif self.handle_first_timestep == "zero":
                # Zero out previous observation
                result["prev_observation"] = {
                    k: np.zeros_like(v) if isinstance(v, np.ndarray) else v for k, v in data.items()
                }
                # Zero previous action, preserving action horizon shape if present
                if "actions" in data and len(data["actions"].shape) == 2:
                    result["prev_actions"] = np.zeros_like(data["actions"])
                else:
                    result["prev_actions"] = np.zeros(self.action_dim)

            elif self.handle_first_timestep == "skip":
                # Mark this sample to be skipped
                result["skip_sample"] = True

        return result


@dataclasses.dataclass(frozen=True)
class LiberoObsDistillationInputs(transforms.DataTransformFn):
    """Transform Libero data for observation distillation training."""

    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0_FAST

    def __call__(self, data: dict) -> dict:
        """Transform data for observation distillation.

        Creates two sets of inputs:
        1. Student inputs: (o_{t-1}, a_{t-1})
        2. Teacher inputs: o_t
        Both predict: a_t
        """
        mask_padding = self.model_type == _model.ModelType.PI0

        # Process current observation (for teacher)
        current_state = transforms.pad_to_dim(
            data.get("state", data.get("observation/state", np.zeros(8))), self.action_dim
        )
        current_base_image = _parse_image(data.get("image", data.get("observation/image", np.zeros((224, 224, 3)))))
        current_wrist_image = _parse_image(
            data.get("wrist_image", data.get("observation/wrist_image", np.zeros((224, 224, 3))))
        )

        # Process previous observation (for student) if available
        if "prev_observation" in data:
            prev_data = data["prev_observation"]
            prev_state = transforms.pad_to_dim(
                prev_data.get("state", prev_data.get("observation/state", np.zeros(8))), self.action_dim
            )
            prev_base_image = _parse_image(
                prev_data.get("image", prev_data.get("observation/image", np.zeros((224, 224, 3))))
            )
            prev_wrist_image = _parse_image(
                prev_data.get("wrist_image", prev_data.get("observation/wrist_image", np.zeros((224, 224, 3))))
            )
        else:
            # If no previous observation, duplicate current (will be handled by transform)
            prev_state = current_state
            prev_base_image = current_base_image
            prev_wrist_image = current_wrist_image

        # Create inputs dict with both current and previous observations
        inputs = {
            # Current observation (for teacher model)
            "state": current_state,
            "image": {
                "base_0_rgb": current_base_image,
                "left_wrist_0_rgb": current_wrist_image,
                "right_wrist_0_rgb": np.zeros_like(current_base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_ if mask_padding else np.True_,
            },
            # Previous observation (for student model)
            "prev_state": prev_state,
            "prev_image": {
                "base_0_rgb": prev_base_image,
                "left_wrist_0_rgb": prev_wrist_image,
                "right_wrist_0_rgb": np.zeros_like(prev_base_image),
            },
            "prev_image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_ if mask_padding else np.True_,
            },
        }

        # Add previous actions if available (for student model)
        if "prev_actions" in data:
            inputs["prev_actions"] = transforms.pad_to_dim(data["prev_actions"], self.action_dim)
        else:
            inputs["prev_actions"] = np.zeros(self.action_dim)

        # Current actions are the target
        if "actions" in data:
            inputs["actions"] = transforms.pad_to_dim(data["actions"], self.action_dim)

        # Pass prompt if available
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoObsDistillationOutputs(transforms.DataTransformFn):
    """Output transform for observation distillation."""

    action_dim: int = 7

    def __call__(self, data: dict) -> dict:
        """Extract actions from model output."""
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
