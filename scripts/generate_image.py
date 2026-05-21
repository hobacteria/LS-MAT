from scripts.models import load_vae_gan, load_unet, load_control_net
from scripts.utils import inference_sample, flat_dataloader, build_flat_samples, cleanup_DDP_checkpoint, parse_train_files
from scripts.registration import inverse_transforms
import nibabel as nib
import os
import shutil
import torch
from monai.networks.schedulers import PNDMScheduler
import monai
import numpy as np
from tqdm import tqdm


def decode(latent, autoencoder):
    with torch.no_grad(), torch.amp.autocast("cuda"):
        image = torch.zeros((latent.shape[0], 1, 256, 256, 256), device=latent.device)
        for i in range(latent.shape[0]):
            image[i] = autoencoder.decode_stage_2_outputs(latent[i].unsqueeze(0))
    return image


def save_image_pair(t1_vol, t2_vol, save_path, age_val, original_spatial_size, args):
    """Save a single T1/T2 pair. t1_vol and t2_vol are [1, H, W, D] tensors."""
    t1_path = os.path.join(save_path, f"T1w_age_{age_val}.nii.gz")
    t2_path = os.path.join(save_path, f"T2w_age_{age_val}.nii.gz")

    if original_spatial_size is not None:
        resize = monai.transforms.Resized(keys=["image"], spatial_size=original_spatial_size, mode="trilinear")
        t1_np = resize({"image": t1_vol.cpu()})["image"][0].numpy().astype(np.float32)
        t2_np = resize({"image": t2_vol.cpu()})["image"][0].numpy().astype(np.float32)
    else:
        t1_np = t1_vol[0].cpu().numpy().astype(np.float32)
        t2_np = t2_vol[0].cpu().numpy().astype(np.float32)

    nib.save(nib.Nifti1Image(t1_np, np.eye(4)), t1_path)
    nib.save(nib.Nifti1Image(t2_np, np.eye(4)), t2_path)

    if args.registration:
        subject_id = save_path.replace(args.output_dir, "").replace("_reg", "")
        inv1 = inverse_transforms({"image_save": t1_path, "subject_id": subject_id, "path_copy": t1_path})["inverse_image"]
        shutil.move(inv1, t1_path)
        inv2 = inverse_transforms({"image_save": t2_path, "subject_id": subject_id, "path_copy": t2_path})["inverse_image"]
        shutil.move(inv2, t2_path)


def _subject_save_path(image_path, args):
    subject_id = os.path.normpath(image_path.replace(args.input_dir, "").replace(".nii.gz", "")).lstrip(os.sep)
    return os.path.join(args.output_dir, subject_id)


def filter_completed(flat_samples, args):
    """Remove (subject, synth_age) pairs whose output files already exist."""
    remaining = []
    for sample in flat_samples:
        save_path = _subject_save_path(sample["image_path"], args)
        age_val = sample["synth_age"]
        t1_done = os.path.exists(os.path.join(save_path, f"T1w_age_{age_val}.nii.gz"))
        t2_done = os.path.exists(os.path.join(save_path, f"T2w_age_{age_val}.nii.gz"))
        if not (t1_done and t2_done):
            remaining.append(sample)
    return remaining


def probe_max_batch_size(args, autoencoder, controlnet, unet, probe_sample):
    """Run a 3-step dry-run with batch_size=1 to measure per-sample peak VRAM."""
    device = args.device
    gpu_utilization = getattr(args, "gpu_utilization", 0.7)

    probe_dl = flat_dataloader([probe_sample], batch_size=1, device=device, cache_rate=0.0, num_workers=0)
    probe_scheduler = PNDMScheduler(num_train_timesteps=1000)

    torch.cuda.reset_peak_memory_stats(device)
    mem_baseline = torch.cuda.memory_allocated(device)

    with torch.no_grad():
        for batch in probe_dl:
            image = batch["image"].to(device)
            sex_tensor = batch["sex"].to(device)
            cross_modality_tensor = 1 - batch["modality"].to(device)
            age_tensor = batch["synth_age"].to(device) / 100

            inference_sample(
                sex_tensor, cross_modality_tensor, age_tensor,
                controlnet, unet, image, probe_scheduler,
                inference_step=10, device=device,
                age_control=True,
            )
            break

    peak_mem = torch.cuda.max_memory_allocated(device)
    per_sample_mem = max(peak_mem - mem_baseline, 1)

    total_vram = torch.cuda.get_device_properties(device).total_memory
    available = total_vram * gpu_utilization - mem_baseline
    max_batch_size = max(1, int(available / per_sample_mem))

    print(
        f"VRAM probe: total={total_vram/1e9:.1f}GB  models={mem_baseline/1e9:.2f}GB  "
        f"per_sample={per_sample_mem/1e9:.2f}GB  → max_batch_size={max_batch_size}"
    )
    return max_batch_size


def generate_image(args):
    autoencoder = load_vae_gan().to(args.device).eval()
    controlnet = load_control_net().to(args.device).eval()
    unet = load_unet().to(args.device).eval()

    autoencoder = cleanup_DDP_checkpoint(autoencoder, args.trained_vae_gan_path)
    unet = cleanup_DDP_checkpoint(unet, args.trained_diffusion_path)
    controlnet = cleanup_DDP_checkpoint(controlnet, args.trained_controlnet_path)

    train_files = parse_train_files(args.input_dir + args.subjects_info)
    flat_samples = build_flat_samples(train_files)
    flat_samples = filter_completed(flat_samples, args)

    if not flat_samples:
        print("All samples already completed.")
        return args

    max_batch_size = probe_max_batch_size(args, autoencoder, controlnet, unet, flat_samples[0])

    infer_dl = flat_dataloader(flat_samples, batch_size=max_batch_size, device=args.device, cache_rate=0.0, num_workers=10)
    noise_scheduler = PNDMScheduler(num_train_timesteps=1000)
    scale_norm = monai.transforms.ScaleIntensityRangePercentilesd(
        keys=["image"], lower=0.0, upper=99.5, b_min=0.0, b_max=1, clip=False
    )

    with torch.no_grad():
        with tqdm(infer_dl) as pbar:
            for batch in pbar:
                B = batch["image"].shape[0]
                pbar.set_description(f"Processing batch of {B} sample(s)")

                image = batch["image"].to(args.device)
                sex_tensor = batch["sex"].to(args.device)
                modality_tensor = batch["modality"].to(args.device)
                cross_modality_tensor = 1 - modality_tensor
                age_tensor = batch["synth_age"].to(args.device) / 100
                original_age = batch["age"].to(args.device)

                # Per-sample age_control: True when synth_age differs from original age
                age_control_mask = (
                    batch["synth_age"].flatten().to(args.device) != original_age.flatten()
                )

                # Step 1: cross-modal synthesis
                # Use age guidance if at least one sample in the batch needs it
                step1_age_control = age_control_mask.any().item()
                latent_cross = inference_sample(
                    sex_tensor, cross_modality_tensor, age_tensor,
                    controlnet, unet, image, noise_scheduler,
                    inference_step=args.inference_step, device=args.device,
                    age_control=step1_age_control,
                )
                cross_output = decode(latent_cross, autoencoder)
                cross_output = torch.clamp(cross_output, min=0)
                cross_output = scale_norm({"image": cross_output})["image"]

                # Step 2 (Option A): run same-modal re-synthesis only when any sample needs it
                if age_control_mask.any():
                    latent_same = inference_sample(
                        sex_tensor, modality_tensor, age_tensor,
                        controlnet, unet, cross_output, noise_scheduler,
                        inference_step=args.inference_step, device=args.device,
                        age_control=False,
                    )
                    same_output = decode(latent_same, autoencoder)
                    same_output = torch.clamp(same_output, min=0)
                    same_output = scale_norm({"image": same_output})["image"]
                    # Samples that don't need age control: use original image directly
                    no_age_ctrl = ~age_control_mask
                    if no_age_ctrl.any():
                        same_output[no_age_ctrl] = image[no_age_ctrl]
                else:
                    same_output = image

                # Brain mask from skull-stripped input
                brain_mask = (image > 0).float()
                cross_output = cross_output * brain_mask
                same_output = same_output * brain_mask

                # Save each sample in the batch individually
                for i in range(B):
                    modality_i = batch["modality_str"][i]
                    if modality_i == "t1":
                        t1_vol, t2_vol = same_output[i, 0:1], cross_output[i, 0:1]
                    elif modality_i == "t2":
                        t1_vol, t2_vol = cross_output[i, 0:1], same_output[i, 0:1]
                    else:
                        raise ValueError(f"Unknown modality: {modality_i}")

                    image_path = batch["image_path"][i]
                    save_path = _subject_save_path(image_path, args)
                    os.makedirs(save_path, exist_ok=True)

                    original_spatial_size = None
                    if not args.registration:
                        original_spatial_size = nib.load(image_path).shape[:3]

                    synth_age_val = batch["synth_age"][i].item()
                    save_image_pair(t1_vol, t2_vol, save_path, synth_age_val, original_spatial_size, args)

    return args
