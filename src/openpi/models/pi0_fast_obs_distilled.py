"""Pi0-FAST model with observation distillation using KL divergence.

This model minimizes KL divergence between:
- Teacher: model(o_t) → a_t (uses current observation)
- Student: model(o_{t-1}, a_{t-1}) → a_t (uses previous observation and action)
"""

import dataclasses
import logging
from typing import Any, Optional

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

import openpi.models.gemma_fast as _gemma
import openpi.models.siglip as _siglip
import openpi.shared.nnx_utils as nnx_utils
from openpi.models import model as _model
from openpi.models.pi0_fast import Pi0FAST, Pi0FASTConfig
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")

PALIGEMMA_EOS_TOKEN = 1


def make_attn_mask(input_mask, mask_ar):
    """Creates attention mask for the model."""
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@jax.vmap
def left_to_right_align(x, input_mask, attn_mask):
    """Converts input from left-align to right-aligned."""
    assert x.ndim == 2
    assert input_mask.ndim == 1
    assert attn_mask.ndim == 2
    assert x.shape[0] == input_mask.shape[0]
    assert attn_mask.shape[0] == attn_mask.shape[1], attn_mask.shape
    seqlen = jnp.max(input_mask * jnp.arange(input_mask.shape[0])) + 1
    x = jnp.roll(x, -seqlen, axis=0)
    input_mask = jnp.roll(input_mask, -seqlen, axis=0)
    attn_mask = jnp.roll(attn_mask, -seqlen, axis=(0, 1))
    return x, input_mask, attn_mask


def put_along_last_axis(arr, indices, values):
    """Like np.put_along_axis(..., axis=-1), since jax is missing it."""
    assert arr.ndim == indices.ndim == values.ndim, (arr.ndim, indices.ndim, values.ndim)
    onehot = jax.nn.one_hot(indices, arr.shape[-1], dtype=values.dtype)
    put_mask = jnp.einsum("...i,...in->...n", jnp.ones(values.shape, jnp.int32), onehot)
    put_values = jnp.einsum("...i,...in->...n", values, onehot)
    return jnp.where(put_mask, put_values, arr)


@dataclasses.dataclass(frozen=True)
class Pi0FASTObsDistilledConfig(_model.BaseModelConfig):
    """Configuration for Pi0-FAST with observation distillation using KL divergence."""

    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"

    # KL distillation configuration
    kl_weight: float = 1.0  # Weight for KL loss term
    teacher_temperature: float = 1.0  # Temperature for softmax in KL computation
    use_teacher_model: bool = True  # Whether to use a separate teacher model
    teacher_checkpoint: str | None = None  # Path to teacher model checkpoint

    # Previous action embedding
    use_prev_action_input: bool = True
    prev_action_embed_dim: int | None = None  # If None, uses paligemma width

    # Set the model specific defaults
    action_dim: int = 32
    action_horizon: int = 32
    max_token_len: int = 250

    # Tokenizer for the fast model
    fast_model_tokenizer: Optional[Any] = None
    fast_model_tokenizer_kwargs: Optional[dict[str, Any]] = None

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0_FAST

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0FASTObsDistilled":
        return Pi0FASTObsDistilled(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "base_1_rgb": image_spec,
                    "wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "base_1_rgb": image_mask_spec,
                    "wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                token_ar_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                token_loss_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.bool_),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        if "lora" in self.paligemma_variant:
            return nnx.All(nnx_utils.PathRegex(".*llm.*"), nnx.Not(nnx_utils.PathRegex(".*lora.*")))
        return nnx.Nothing


class Pi0FASTObsDistilled(_model.BaseModel):
    """Pi0-FAST model with observation distillation using KL divergence.

    Student model: Learns to predict a_t from (o_{t-1}, a_{t-1})
    Teacher model: Predicts a_t from o_t
    Training: Minimizes KL divergence between student and teacher predictions
    """

    def __init__(self, config: Pi0FASTObsDistilledConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.config = config
        paligemma_config = _gemma.get_config(config.paligemma_variant)

        # Initialize the student model (uses previous obs and action)
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                **paligemma_config,
                embed_dtype=config.dtype,
                cache_dtype=config.dtype,
            )
        )
        llm.lazy_init(rngs=rngs, method="init")

        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        # Add projection layer for previous action (a_{t-1})
        embed_dim = config.prev_action_embed_dim or paligemma_config.width
        self.prev_action_proj = nnx.Linear(config.action_dim, embed_dim, rngs=rngs)

        # Initialize teacher model if needed (uses current obs)
        self.teacher_model = None
        if config.use_teacher_model and config.teacher_checkpoint:
            # Only create separate teacher model if checkpoint is provided
            # Create a standard Pi0FAST model as teacher
            teacher_config = Pi0FASTConfig(
                action_dim=config.action_dim,
                action_horizon=config.action_horizon,
                max_token_len=config.max_token_len,
                paligemma_variant=config.paligemma_variant,
                dtype=config.dtype,
            )
            self.teacher_model = Pi0FAST(teacher_config, rngs=rngs)
            logger.info(f"Loading teacher model from {config.teacher_checkpoint}")
            # TODO: Implement checkpoint loading
        elif config.use_teacher_model:
            # No teacher checkpoint provided - will use self-distillation
            logger.info("No teacher checkpoint provided - using self-distillation")
            self.teacher_model = None

    def embed_inputs(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Int[at.Array, "b s"]]:
        """Embed inputs without action (for self-distillation teacher path)."""
        input_mask = []
        ar_mask = []
        token_embeddings = []

        # Embed images from observation
        for name in obs.images:
            image_token_embeddings, _ = self.PaliGemma.img(obs.images[name], train=False)

            token_embeddings.append(image_token_embeddings)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_token_embeddings.shape[1],
                )
            )
            ar_mask.append(0 * input_mask[-1])

        # Add tokenized inputs (prompt + state)
        assert obs.tokenized_prompt is not None, "Tokenized prompt is required"
        assert obs.tokenized_prompt_mask is not None, "Tokenized prompt mask is required"
        assert obs.token_ar_mask is not None, "Token auto-regressive mask is required"

        tokenized_inputs_embeddings = self.PaliGemma.llm(obs.tokenized_prompt, embed_only=True)
        token_embeddings.append(tokenized_inputs_embeddings)
        input_mask.append(obs.tokenized_prompt_mask)
        ar_mask.append(obs.token_ar_mask)

        return (
            jnp.concatenate(token_embeddings, axis=1),
            jnp.concatenate(input_mask, axis=1),
            jnp.concatenate(ar_mask, axis=1),
        )

    @at.typecheck
    def embed_inputs_with_prev_action(
        self, obs: _model.Observation, prev_actions: at.Float[at.Array, "b ah ad"] | None = None
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Int[at.Array, "b s"]]:
        """Embed inputs including previous action horizon (a_{t-1}) for student model."""
        input_mask = []
        ar_mask = []
        token_embeddings = []

        # Embed images from o_{t-1} (previous observation)
        for name in obs.images:
            image_token_embeddings, _ = self.PaliGemma.img(obs.images[name], train=False)

            token_embeddings.append(image_token_embeddings)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_token_embeddings.shape[1],
                )
            )
            ar_mask.append(0 * input_mask[-1])

        # Add previous action horizon embedding if provided (a_{t-1})
        if prev_actions is not None and self.config.use_prev_action_input:
            # Previous actions shape: [batch_size, action_horizon, action_dim]
            batch_size, action_horizon, action_dim = prev_actions.shape

            # Project each action in the horizon to embedding dimension
            # Reshape to [batch_size * action_horizon, action_dim] for projection
            prev_actions_flat = einops.rearrange(prev_actions, "b ah ad -> (b ah) ad")
            prev_action_emb_flat = self.prev_action_proj(prev_actions_flat)

            # Reshape back to [batch_size, action_horizon, embed_dim]
            prev_action_emb = einops.rearrange(
                prev_action_emb_flat, "(b ah) e -> b ah e", b=batch_size, ah=action_horizon
            )

            token_embeddings.append(prev_action_emb)
            input_mask.append(jnp.ones((batch_size, action_horizon), dtype=jnp.bool_))
            ar_mask.append(jnp.zeros((batch_size, action_horizon), dtype=jnp.int32))

        # Add tokenized inputs (prompt + state from o_{t-1})
        assert obs.tokenized_prompt is not None, "Tokenized prompt is required"
        assert obs.tokenized_prompt_mask is not None, "Tokenized prompt mask is required"
        assert obs.token_ar_mask is not None, "Token auto-regressive mask is required"

        tokenized_inputs_embeddings = self.PaliGemma.llm(obs.tokenized_prompt, embed_only=True)
        token_embeddings.append(tokenized_inputs_embeddings)
        input_mask.append(obs.tokenized_prompt_mask)
        ar_mask.append(obs.token_ar_mask)

        return (
            jnp.concatenate(token_embeddings, axis=1),
            jnp.concatenate(input_mask, axis=1),
            jnp.concatenate(ar_mask, axis=1),
        )

    def compute_student_logits(
        self,
        prev_observation: _model.Observation,
        prev_actions: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> at.Float[at.Array, "b s v"]:
        """Compute student model logits from (o_{t-1}, a_{t-1})."""
        # Embed inputs with previous action
        token_embeddings, input_mask, ar_mask = self.embed_inputs_with_prev_action(prev_observation, prev_actions)

        # Create attention mask
        attn_mask = make_attn_mask(input_mask, ar_mask)

        # Forward pass through language model
        targets_shape = prev_observation.tokenized_prompt.shape[1] - 1
        pre_logits, _, _ = self.PaliGemma.llm(
            embedded_prefix=token_embeddings[:, :-1],
            mask=attn_mask[:, :-1, :-1],
            return_prelogits=True,
        )

        # Decode logits
        logits, _ = self.PaliGemma.llm(
            pre_logits=pre_logits[:, -targets_shape:],
        )

        return logits

    def compute_teacher_logits(
        self,
        current_observation: _model.Observation,
    ) -> at.Float[at.Array, "b s v"]:
        """Compute teacher model logits from o_t.

        If teacher_model is None, use the student model itself as teacher
        (self-distillation using current observation without previous action).
        """
        if self.teacher_model is None:
            # Self-distillation: use student model with current observation (no prev action)
            # This gives us logits from o_t only, without previous action conditioning
            token_embeddings, input_mask, ar_mask = self.embed_inputs(current_observation)
        else:
            # Use separate teacher model's standard embedding (without previous action)
            token_embeddings, input_mask, ar_mask = self.teacher_model.embed_inputs(current_observation)

        # Create attention mask
        attn_mask = make_attn_mask(input_mask, ar_mask)

        # Forward pass through the appropriate model
        targets_shape = current_observation.tokenized_prompt.shape[1] - 1

        if self.teacher_model is None:
            # Use student model (self) for forward pass
            pre_logits, _, _ = self.PaliGemma.llm(
                embedded_prefix=token_embeddings[:, :-1],
                mask=attn_mask[:, :-1, :-1],
                return_prelogits=True,
            )

            # Decode logits
            logits, _ = self.PaliGemma.llm(
                pre_logits=pre_logits[:, -targets_shape:],
            )
        else:
            # Use teacher model for forward pass
            pre_logits, _, _ = self.teacher_model.PaliGemma.llm(
                embedded_prefix=token_embeddings[:, :-1],
                mask=attn_mask[:, :-1, :-1],
                return_prelogits=True,
            )

            # Decode logits
            logits, _ = self.teacher_model.PaliGemma.llm(
                pre_logits=pre_logits[:, -targets_shape:],
            )

        return logits

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        prev_observation: _model.Observation | None = None,
        prev_actions: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> at.Float[at.Array, "*b"]:
        """Compute loss with KL distillation.

        Args:
            rng: Random key
            observation: Current observation o_t (for teacher)
            actions: Current action horizon a_t (target)
            train: Whether in training mode
            prev_observation: Previous observation o_{t-1} (for student)
            prev_actions: Previous action horizon a_{t-1} (for student)

        Returns:
            Combined loss (CE + KL divergence)
        """
        # Use previous observation for student if provided
        student_obs = prev_observation if prev_observation is not None else observation

        # Preprocess observations
        student_obs = _model.preprocess_observation(
            rng if train else None, student_obs, train=train, image_keys=list(student_obs.images.keys())
        )

        # Compute student logits from (o_{t-1}, a_{t-1})
        student_logits = self.compute_student_logits(student_obs, prev_actions)

        # Compute cross-entropy loss for student
        targets = jax.nn.one_hot(
            student_obs.tokenized_prompt[:, 1:],
            self.PaliGemma.llm.module.vocab_size,
        )
        student_logp = jax.nn.log_softmax(student_logits, axis=-1)

        assert student_obs.token_loss_mask is not None, "Token loss mask is required"
        loss_mask = student_obs.token_loss_mask[:, 1:]

        # CE loss
        ce_loss = -jnp.sum(targets * student_logp, axis=-1)
        ce_loss = jnp.sum(ce_loss * loss_mask, axis=-1) / jnp.clip(jnp.sum(loss_mask, -1), 1)

        # Compute KL divergence if teacher model is enabled
        # (can be self-distillation if teacher_model is None)
        if self.config.use_teacher_model:
            # Preprocess current observation for teacher
            teacher_obs = _model.preprocess_observation(
                None,
                observation,
                train=False,  # Teacher doesn't need augmentation
                image_keys=list(observation.images.keys()),
            )

            # Get teacher logits from o_t
            # If teacher_model is None, this will use self-distillation
            teacher_logits = jax.lax.stop_gradient(self.compute_teacher_logits(teacher_obs))

            # Compute KL divergence
            student_probs = jax.nn.softmax(student_logits / self.config.teacher_temperature, axis=-1)
            teacher_probs = jax.nn.softmax(
                teacher_logits / self.config.teacher_temperature, axis=-1
            )

            # KL(teacher || student) = sum(teacher * log(teacher/student))
            kl_div = teacher_probs * (jnp.log(teacher_probs + 1e-8) - jnp.log(student_probs + 1e-8))
            kl_div = jnp.sum(kl_div, axis=-1)  # Sum over vocab

            # Apply mask and average
            kl_loss = jnp.sum(kl_div * loss_mask, axis=-1) / jnp.clip(jnp.sum(loss_mask, -1), 1)

            # Combine losses
            total_loss = ce_loss + self.config.kl_weight * kl_loss
        else:
            total_loss = ce_loss

        return total_loss

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        prev_actions: at.Float[at.Array, "b ah ad"] | None = None,
        max_decoding_steps: int | at.Int[at.Array, ""] = 256,
        temperature: float = 0.0,
    ) -> _model.Actions:
        """Sample actions using previous observation and action (student model).

        During inference, the student model uses (o_{t-1}, a_{t-1}) to predict a_t.
        """
        observation = _model.preprocess_observation(
            None, observation, train=False, image_keys=list(observation.images.keys())
        )

        # Embed inputs with previous action
        prefix_token_embeddings, prefix_mask, prefix_ar_mask = self.embed_inputs_with_prev_action(
            observation, prev_actions
        )
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)

        # Left to right align
        prefix_token_embeddings, prefix_mask, prefix_attn_mask = left_to_right_align(
            prefix_token_embeddings, prefix_mask, prefix_attn_mask
        )
        prefill_size = prefix_token_embeddings.shape[1]
        prefill_len = jnp.sum(prefix_mask, axis=-1)
        prefix_start = prefill_size - prefill_len

        # Fill KV cache
        prefix_attn_mask = jnp.pad(prefix_attn_mask, ((0, 0), (0, 0), (0, max_decoding_steps)))
        prefix_positions = jnp.cumsum(prefix_mask, axis=-1) - 1
        prefix_logits, kv_cache, _ = self.PaliGemma.llm(
            embedded_prefix=prefix_token_embeddings, mask=prefix_attn_mask, positions=prefix_positions, decode=True
        )

        # Prepare decoding
        last_logit = prefix_logits[:, -1:]
        output_tokens = jnp.zeros((last_logit.shape[0], max_decoding_steps))

        def step(carry):
            rng, last_logit, output_tokens, cache, _, step = carry

            # Sample token
            rng, rng_step = jax.random.split(rng)
            token = jax.lax.cond(
                temperature > 0.0,
                lambda _: jax.random.categorical(rng_step, last_logit / temperature, axis=-1),
                lambda _: jnp.argmax(last_logit, axis=-1),
                operand=None,
            )
            output_tokens = put_along_last_axis(output_tokens, jnp.broadcast_to(step, (token.shape[0], 1)), token)

            # Check for early stopping
            has_eos = jnp.any(token == PALIGEMMA_EOS_TOKEN, axis=-1)
            all_eos = jnp.all(has_eos)

            # Decode one step
            token_embedding = self.PaliGemma.llm(token, embed_only=True)
            positions = prefill_len[:, None] + step + 1
            mask = jnp.logical_and(
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :] >= prefix_start[:, None, None],
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :]
                < (jnp.broadcast_to(prefill_size + step + 1, (prefix_start.shape[0], 1, 1))),
            )
            last_logit, kv_cache, _ = self.PaliGemma.llm(
                embedded_prefix=token_embedding, mask=mask, positions=positions, decode=True, kv_cache=cache
            )

            return rng, last_logit, output_tokens, kv_cache, all_eos, step + 1

        def cond(carry):
            _, _, _, _, all_eos, step = carry
            return (~all_eos) & (step < max_decoding_steps)

        # Decode loop
        _, _, output_tokens, _, _, _ = jax.lax.while_loop(
            cond, step, (rng, last_logit, output_tokens, kv_cache, False, 0)
        )
        return output_tokens
