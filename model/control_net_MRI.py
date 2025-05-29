# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from model.control_net import ControlNet
from monai.networks.nets.diffusion_model_unet import get_timestep_embedding
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi

class ControlNetMRI(ControlNet):
    """
    Control network for diffusion models based on Zhang and Agrawala "Adding Conditional Control to Text-to-Image
    Diffusion Models" (https://arxiv.org/abs/2302.05543)

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        num_res_blocks: number of residual blocks (see ResnetBlock) per level.
        num_channels: tuple of block output channels.
        attention_levels: list of levels to add attention.
        norm_num_groups: number of groups for the normalization.
        norm_eps: epsilon for the normalization.
        resblock_updown: if True use residual blocks for up/downsampling.
        num_head_channels: number of channels in each attention head.
        with_conditioning: if True add spatial transformers to perform conditioning.
        transformer_num_layers: number of layers of Transformer blocks to use.
        cross_attention_dim: number of context dimensions to use.
        num_class_embeds: if specified (as an int), then this model will be class-conditional with `num_class_embeds`
            classes.
        upcast_attention: if True, upcast attention operations to full precision.
        conditioning_embedding_in_channels: number of input channels for the conditioning embedding.
        conditioning_embedding_num_channels: number of channels for the blocks in the conditioning embedding.
        use_checkpointing: if True, use activation checkpointing to save memory.
        include_fc: whether to include the final linear layer. Default to False.
        use_combined_linear: whether to use a single linear layer for qkv projection, default to False.
        use_flash_attention: if True, use flash attention for a memory efficient attention mechanism.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        num_res_blocks: Sequence[int] | int = (2, 2, 2, 2),
        num_channels: Sequence[int] = (32, 64, 64, 64),
        attention_levels: Sequence[bool] = (False, False, True, True),
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        resblock_updown: bool = False,
        num_head_channels: int | Sequence[int] = 8,
        with_conditioning: bool = False,
        transformer_num_layers: int = 1,
        cross_attention_dim: int | None = None,
        num_class_embeds: int | None = None,
        upcast_attention: bool = False,
        conditioning_embedding_in_channels: int = 1,
        conditioning_embedding_num_channels: Sequence[int] = (16, 32, 96, 256),
        use_checkpointing: bool = True,
        include_fc: bool = False,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
        include_sub_cat_embed: bool = False,
        include_sex_embed: bool = False,
        include_modality_embed: bool = False,
        include_age_embed: bool = False
        
    ) -> None:
        # time
        time_embed_dim = num_channels[0] * 4
        self.new_time_embed_dim = time_embed_dim * (1+include_sub_cat_embed + include_sex_embed+ include_modality_embed+include_age_embed*4)
        super().__init__(
            spatial_dims,
            in_channels,
            num_res_blocks,
            num_channels,
            attention_levels,
            norm_num_groups,
            norm_eps,
            resblock_updown,
            num_head_channels,
            with_conditioning,
            transformer_num_layers,
            cross_attention_dim,
            num_class_embeds,
            upcast_attention,
            conditioning_embedding_in_channels,
            conditioning_embedding_num_channels,
            include_fc,
            use_combined_linear,
            use_flash_attention,
        )
        self.use_checkpointing = use_checkpointing
        
        self.time_embed = self._create_embedding_module(num_channels[0], time_embed_dim)

        # class embedding
        self.num_class_embeds = num_class_embeds
        if num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)

        self.include_sub_cat_embed = include_sub_cat_embed
        self.include_sex_embed = include_sex_embed
        self.include_modality_embed = include_modality_embed
        self.include_age_embed = include_age_embed
        
        
        new_time_embed_dim = time_embed_dim
        if self.include_sub_cat_embed:
            self.sub_cat_embed_layer = self._create_embedding_module(3, time_embed_dim)
            new_time_embed_dim += time_embed_dim
        if self.include_sex_embed:
            self.sex_embed_layer = self._create_embedding_module(2, time_embed_dim)
            new_time_embed_dim += time_embed_dim
        if self.include_modality_embed:
            self.modality_embed_layer = self._create_embedding_module(2, time_embed_dim)
            new_time_embed_dim += time_embed_dim
        if self.include_age_embed:
            self.age_embed_layer = self._create_embedding_module(num_channels[0], time_embed_dim *4) ## it took cosine/sine embeding.
            new_time_embed_dim += time_embed_dim *4
            
    def _create_embedding_module(self, input_dim, embed_dim):
        model = nn.Sequential(nn.Linear(input_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))
        return model
    
    def _get_input_embeddings(self, emb, sub_cat_index, sex_index,modality_index,age_index):
        if self.include_sub_cat_embed:
            _emb = self.sub_cat_embed_layer(sub_cat_index)
            emb = torch.cat((emb, _emb), dim=1)
            
        if self.include_sex_embed:
            _emb = self.sex_embed_layer(sex_index) #####
            emb = torch.cat((emb, _emb), dim=1)
            
        if self.include_modality_embed:
            _emb = self.modality_embed_layer(modality_index)
            emb = torch.cat((emb, _emb), dim=1)
            
        if self.include_age_embed:
            age_embed = get_timestep_embedding(age_index.reshape(-1), self.block_out_channels[0]) ## positional encoding for age.
            _emb = self.age_embed_layer(age_embed)
            emb = torch.cat((emb, _emb), dim=1)
        return emb

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        controlnet_cond: torch.Tensor,
        conditioning_scale: float = 1.0,
        context: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
        sub_cat_index_tensor: torch.Tensor | None = None,
        sex_index_tensor: torch.Tensor | None = None,
        modality_index_tensor: torch.Tensor | None = None,
        age_tensor: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        
        emb = self._prepare_time_and_class_embedding(x, timesteps, class_labels)
        emb = self._get_input_embeddings(emb, sub_cat_index_tensor, sex_index_tensor, modality_index_tensor,age_tensor)
        h = self._apply_initial_convolution(x)
        if self.use_checkpointing:
            controlnet_cond = torch.utils.checkpoint.checkpoint(
                self.controlnet_cond_embedding, controlnet_cond, use_reentrant=False
            )
        else:
            controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)

        h += controlnet_cond
        down_block_res_samples, h = self._apply_down_blocks(emb, context, h)
        h = self._apply_mid_block(emb, context, h)
        down_block_res_samples, mid_block_res_sample = self._apply_controlnet_blocks(h, down_block_res_samples)
        # scaling
        down_block_res_samples = [h * conditioning_scale for h in down_block_res_samples]
        mid_block_res_sample *= conditioning_scale

        return down_block_res_samples, mid_block_res_sample

    def _prepare_time_and_class_embedding(self, x, timesteps, class_labels):
        # 1. time
        t_emb = get_timestep_embedding(timesteps, self.block_out_channels[0])

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=x.dtype)
        emb = self.time_embed(t_emb)

        # 2. class
        if self.num_class_embeds is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")
            class_emb = self.class_embedding(class_labels)
            class_emb = class_emb.to(dtype=x.dtype)
            emb = emb + class_emb

        return emb

    def _apply_initial_convolution(self, x):
        # 3. initial convolution
        h = self.conv_in(x)
        return h

    def _apply_down_blocks(self, emb, context, h):
        # 4. down
        if context is not None and self.with_conditioning is False:
            raise ValueError("model should have with_conditioning = True if context is provided")
        down_block_res_samples: list[torch.Tensor] = [h]
        for downsample_block in self.down_blocks:
            h, res_samples = downsample_block(hidden_states=h, temb=emb, context=context)
            for residual in res_samples:
                down_block_res_samples.append(residual)

        return down_block_res_samples, h

    def _apply_mid_block(self, emb, context, h):
        # 5. mid
        h = self.middle_block(hidden_states=h, temb=emb, context=context)
        return h

    def _apply_controlnet_blocks(self, h, down_block_res_samples):
        # 6. Control net blocks
        controlnet_down_block_res_samples = []
        for down_block_res_sample, controlnet_block in zip(down_block_res_samples, self.controlnet_down_blocks):
            down_block_res_sample = controlnet_block(down_block_res_sample)
            controlnet_down_block_res_samples.append(down_block_res_sample)

        mid_block_res_sample = self.controlnet_mid_block(h)

        return controlnet_down_block_res_samples, mid_block_res_sample
