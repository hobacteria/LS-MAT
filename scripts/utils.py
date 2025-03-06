from tqdm import tqdm
import torch
import monai
import numpy as np
import json
from types import SimpleNamespace

from monai.transforms import Compose

from model.diffusion_model_unet_MRI import DiffusionModelUNetMRI
from model.control_net_MRI import ControlNetMRI
from monai.data import ThreadDataLoader,DistributedSampler

def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return SimpleNamespace(**config)

def initialize_noise_latents(latent_shape, batch_size,device):
    """
    Initialize random noise latents for image generation with float16.

    Args:
        latent_shape (tuple): The shape of the latent space.
        device (torch.device): The device to create the tensor on.

    Returns:
        torch.Tensor: Initialized noise latents.
    """
    return (
        torch.randn(
            [
                batch_size,
            ]
            + list(latent_shape)
        )
        .half()
        .to(device)
    )
    
def inference_sample(sex_index_tensor,modality_index_tensor,age_tensor, controlnet, unet,condition_sample, noise_scheduler,inference_step,device):
    
    noise_scheduler.set_timesteps(inference_step)
    init_noise = initialize_noise_latents([3,64, 64, 64],condition_sample.shape[0], device).float()
    noisy_latent = init_noise.clone()
    
    if isinstance(controlnet,type(None)) & isinstance(condition_sample, type(None)):
        sampler = unet_process_sample
    elif isinstance(controlnet,ControlNetMRI) & isinstance(condition_sample, torch.Tensor):
        sampler = controlnet_process_sample
    else:
        raise ValueError("controlnet and conditioning_sample should be None or torch.Tensor")
    for t in noise_scheduler.timesteps:
        
        time_step = torch.tensor([t],dtype = float).to(device).repeat(condition_sample.shape[0])
        noise_pred = sampler(noisy_latent, condition_sample, time_step,sex_index_tensor,modality_index_tensor,age_tensor, controlnet, unet,device)
        noisy_latent, _ = noise_scheduler.step(noise_pred, t, noisy_latent)
        
    return noisy_latent


def unet_process_sample(noisy_latent, conditioning_sample,time_step,sex_index_tensor,modality_index_tensor,age_tensor, controlnet, unet,device):
    
    if not (isinstance(controlnet,None) & isinstance(conditioning_sample, None)):
        raise ValueError("controlnet and conditioning_sample should be None")

    noise_pred_condition = unet(
        x=noisy_latent,
        timesteps=time_step.to(device),
        #sub_cat_index_tensor=sub_cat_index_tensor,
        sex_index_tensor=sex_index_tensor, ## pad for null condition space
        modality_index_tensor=modality_index_tensor,## pad for null condition space
        #age_tensor = age_tensor,
        age_tensor = age_tensor
    )
    return noise_pred_condition


def controlnet_process_sample(noisy_latent, conditioning_sample,time_step,sex_index_tensor,modality_index_tensor,age_tensor, controlnet, unet,device):
    
    # create noisy latent

    # get controlnet output
    down_block_res_samples, mid_block_res_sample = controlnet(
        x=noisy_latent, timesteps=time_step, controlnet_cond=conditioning_sample,
        #sub_cat_index_tensor=sub_cat_index_tensor[:,:-1],
        sex_index_tensor=sex_index_tensor, ## no null condition input in controlnet, reduce last one.
        modality_index_tensor=modality_index_tensor,
        age_tensor = age_tensor,
    )

    # get noise prediction from diffusion unet
    #scales = [0.825 ** float(12 - i) for i in range(13)]
    #down_block_res_samples,mid_block_res_sample = [sample * scales[i] for i,sample in enumerate(down_block_res_samples)], mid_block_res_sample * scales[12]
    down_block_res_samples,mid_block_res_sample = [d for d in down_block_res_samples],mid_block_res_sample
    

    noise_pred_condition = unet(
        x=noisy_latent,
        timesteps=time_step.to(device),
        #sub_cat_index_tensor=sub_cat_index_tensor,
        sex_index_tensor=sex_index_tensor, ## pad for null condition space
        modality_index_tensor=modality_index_tensor,## pad for null condition space
        #age_tensor = age_tensor,
        age_tensor = age_tensor,
        down_block_additional_residuals=down_block_res_samples,
        mid_block_additional_residual=mid_block_res_sample,
    )
    return noise_pred_condition


def cleanup_DDP_checkpoint(model, checkpoint_path):
    """
    'module.' 접두사가 붙은 state_dict를 처리하여 모델에 로드하는 함수.

    Args:
    model (torch.nn.Module): 모델 객체
    checkpoint_path (str): 체크포인트 파일 경로
    
    Returns:
    model: 체크포인트가 로드된 모델
    """
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path,weights_only=False)
    state_dict_key = [key for key in checkpoint.keys() if 'state_dict' in key]
    if len(state_dict_key) > 0:
        checkpoint = checkpoint[state_dict_key[0]]
    # state_dict에서 'module.' 접두사 제거
    new_state_dict = {}

    for key in checkpoint.keys():
        new_key = key.replace('module.', '')  # 'module.' 제거
        new_state_dict[new_key] = checkpoint[key]

    # 새로운 state_dict로 모델의 가중치를 로드
    model.load_state_dict(new_state_dict)
    
    return model

def transformation(image):
    transform = Compose(
        [
            monai.transforms.LoadImaged(keys=["image"],allow_missing_keys=True),
            monai.transforms.EnsureChannelFirstd(keys=["image"],allow_missing_keys=True),
            monai.transforms.Orientationd(keys=["image"], axcodes="RAS"),
            monai.transforms.EnsureTyped(keys=["image"], dtype=torch.float32),
            monai.transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=0.0, upper=99.5, b_min=0.0, b_max=1, clip=False),
            monai.transforms.Resized(keys=["image"], spatial_size=(256,256,256), mode="trilinear")]
    )
    return transform(image)


def parse_train_files(file = 'subjects.txt'):
    """
    학습 데이터 파일을 파싱합니다.
    """
    import re
    import ast
    with open(file, "r") as f:
        lines = f.readlines()
    train_files = []
    for line in lines:
        line = line.strip()
        train_path,modality,sex,age,synth_age = re.split(r',\s*(?![^\[\]]*\])', line) ## ignore comma in list
        synth_age = ast.literal_eval(synth_age) ## change list string to list
        train_files.append({"image": train_path,
                            "image_path": train_path,
                            "modality": modality,
                            "modality_str": modality,
                            "sex" : sex,
                            "age":age,
                            "synth_age":synth_age})
    return train_files



def dataloader(
    train_files: list, device: torch.device, cache_rate: float, num_workers: int = 2, batch_size: int = 1, drop_out=False, rank=None, world_size=None,DDP = False
):
    """
    학습 데이터를 준비합니다.
    """
    
    def get_one_hot_with_dropout(value, mapping, dropout=0.0):
        """
        General function to perform one-hot encoding with optional dropout.

        Args:
            value (str): The input value to encode.
            mapping (dict): A dictionary mapping input values to their indices.
            dropout (float): Dropout probability (0.0 to 1.0).
            use_null_condition (bool): Whether to add a null condition for dropout.

        Returns:
            torch.Tensor: One-hot encoded tensor.
        """
        _length = len(np.unique(list(mapping.values())))
        vector_length = _length + (1 if dropout > 0 else 0)
        one_hot = torch.zeros(vector_length, dtype=torch.float)

        # Dropout logic
        if dropout > 0 and np.random.rand() < dropout:
            one_hot[-1] = 1  # Set the last index for null condition
            return one_hot

        # One-hot encoding
        if value in mapping:
            one_hot[mapping[value]] = 1
            return one_hot
        else:
            print(f"Cannot find in {value}")
            return None

    # Example mappings
    sex_mapping = {'f': 0, 'female': 0, 'm': 1, 'male': 1}
    modality_mapping = {'t1': 0, 't2': 1}

    # Examples of usage
    
    train_transforms = Compose(
        [
            monai.transforms.LoadImaged(keys=["image"]),
            monai.transforms.EnsureChannelFirstd(keys=["image"]),
            monai.transforms.Orientationd(keys=["image"], axcodes="RAS"),
            monai.transforms.EnsureTyped(keys=["image"], dtype=torch.float32),
            monai.transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=0.0, upper=99.5, b_min=0.0, b_max=1, clip=False),
            monai.transforms.Resized(keys=["image"], spatial_size=(256,256,256), mode="trilinear"),
            monai.transforms.Lambdad(keys = ['sex'],func = lambda x: get_one_hot_with_dropout(x, sex_mapping, drop_out).to(device)),
            monai.transforms.Lambdad(keys = ['modality'],func = lambda x: get_one_hot_with_dropout(x, modality_mapping, drop_out).to(device)),
            monai.transforms.Lambdad(keys = ['age'],func = lambda x: torch.Tensor([float(x)]).type(torch.float).to(device)),
            monai.transforms.Lambdad(keys = ['synth_age'],func = lambda x: torch.Tensor(x).reshape(-1).type(torch.float).to(device)),

        ]
    )

    train_ds = monai.data.CacheDataset(
        data=train_files, transform=train_transforms, cache_rate=cache_rate, num_workers=num_workers
    )
    if DDP:
        sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank) if rank is not None else None
    else : sampler = None

    return ThreadDataLoader(train_ds, num_workers=0, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler)

