from model.diffusion_model_unet_MRI import DiffusionModelUNetMRI
from model.control_net_MRI import ControlNetMRI
from model.vae import AutoencoderKlMaisi
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi

def load_unet():
    """
    UNet 모델을 로드합니다.
    """
    unet = DiffusionModelUNetMRI(
        spatial_dims=3,
        in_channels=3,
        out_channels=3,
        num_channels=[128, 256, 384, 512],
        attention_levels=[False, False, True, True],
        num_head_channels=[0, 0, 128, 256],
        num_res_blocks=2,
        use_flash_attention=True,
        #include_sub_cat_embed=True,
        include_sex_embed=True,
        include_modality_embed=True,
        include_age_embed = True,
        cfg = False
    )
    return unet


def load_control_net():
    control_net = ControlNetMRI(
        spatial_dims=3,
        in_channels=3,
        num_channels=[128, 256, 384, 512],
        attention_levels=[False, False, True, True],
        num_head_channels=[0, 0, 128, 256],
        num_res_blocks=2,
        use_flash_attention=True,
        conditioning_embedding_in_channels = 1,
        conditioning_embedding_num_channels = [8, 32, 64],
        #include_sub_cat_embed=True,
        include_sex_embed=True,
        include_modality_embed=True,
        include_age_embed = True
    )
    return control_net

def load_vae_gan():
    vae_gan = AutoencoderKlMaisi(
        spatial_dims=3,
        in_channels=1,
        out_channels = 1,
        latent_channels = 3,
        num_channels = [64, 128, 256],
        num_res_blocks = [2,2,2],
        norm_num_groups = 32,
        norm_eps = 1e-6,
        attention_levels = [False, False, False],
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False,
        use_checkpointing=False,
        use_convtranspose=False,
        norm_float16=True,
        num_splits=8,
        dim_split=1
    )
    return vae_gan
