## make dataloder and encode and to controlnet and save to output images
from scripts.models import load_vae_gan,load_unet,load_control_net
from scripts.utils import inference_sample,dataloader,cleanup_DDP_checkpoint,parse_train_files
from scripts.registration import Inverse_imaged
import nibabel as nib
import os
import torch
from monai.networks.schedulers.pndm import PNDMScheduler
import numpy as np
from tqdm import tqdm
def encode(image,autoencoder):
    latent = autoencoder.encode_stage_2_inputs(image)
    return latent

def decode(latent,autoencoder):
    ## because of vram issue, we decode image separately by batch
    image = torch.zeros((latent.shape[0],1,256,256,256)).to(latent.device)
    for i in range(latent.shape[0]):
        image[i] = autoencoder.decode_stage_2_outputs(latent[i].unsqueeze(0))
    return image

def to_controlnet(image,controlnet):
    output = inference_sample(image, controlnet)
    return output

def save_by_synthesis_age(T1_image,T2_image,save_path,age_tensor,args):
    T1_save_path = os.path.join(save_path,'T1w.nii.gz')
    T2_save_path = os.path.join(save_path,'T2w.nii.gz')
    ## clamp image to 0~
    T1_image = torch.clamp(T1_image,min = 0)
    T2_image = torch.clamp(T2_image,min = 0)
    
    
    for i,age_syn in enumerate(age_tensor):
        T1_age_save_path = T1_save_path.replace('.nii.gz',f'_age_{age_syn.item()}.nii.gz')
        T2_age_save_path = T2_save_path.replace('.nii.gz',f'_age_{age_syn.item()}.nii.gz')
        ## save t1w image
        nib_t1_image = nib.Nifti1Image(T2_image[i,0].cpu().numpy().astype(np.float32), np.eye(4))
        nib.save(nib_t1_image, T1_age_save_path)
        ## save t2w image
        nib_t2_image = nib.Nifti1Image(T1_image[i,0].cpu().numpy().astype(np.float32), np.eye(4))
        nib.save(nib_t2_image, T2_age_save_path)
        
        #if args.registration == 1:
        #    ## after generate image, we need inverse transform to original MRI space
        #    ## inverse transform에서 tmp에 나오는 output이 뭔지 확인하고, inverse 할 때마다 원래 input만 남게 만들기.
        #    ## 연령별로 다 하고 나면 subject id에 해당하는 폴더 삭제하기
        #    subject_id = save_path.replace(args.output_dir,'') ## 이건 맞는지 확인
        #    
        #    invers_transform = Inverse_imaged(keys = 'image')
        #    
        #    inverse_image_1 = invers_transform({'image_save':T1_age_save_path,'subject_id':},save_tmpdir=True)['inverse_image']
        #    os.system(f'mv {inverse_image_1} {T1_age_save_path}')
        #    inverse_image_2 = invers_transform({'image_save':T2_age_save_path},save_tmpdir=True)['inverse_image']
        #    os.system(f'mv {inverse_image_2} {T2_age_save_path}')
        #    os.system(f'rm -r ./.tmp') ## remove tmp folder


def generate_image(args):
    ## load models
    autoencoder = load_vae_gan().to(args.device).eval()
    controlnet = load_control_net().to(args.device).eval()
    unet = load_unet().to(args.device).eval()
    ## cleanup DDP checkpoint, we saved model with DDP
    autoencoder = cleanup_DDP_checkpoint(autoencoder, args.trained_vae_gan_path)
    unet = cleanup_DDP_checkpoint(unet, args.trained_diffusion_path)
    controlnet = cleanup_DDP_checkpoint(controlnet, args.trained_controlnet_path)
    ## make dataloader
    train_files = parse_train_files(args.input_dir + '/subjects.txt')
    infer_dl = dataloader(train_files,batch_size=1,device='cuda',cache_rate=0.0,num_workers=10)
    noise_scheduler = PNDMScheduler(num_train_timesteps = 1000)
    with torch.no_grad(),torch.cuda.amp.autocast():
        with tqdm(infer_dl) as pbar:
            for batch in pbar:
                
                pbar.set_description(f"Processing {batch['image_path'][0]}")
                
                batch['synth_age']= batch['synth_age'].transpose(1,0).to(args.device)
                synth_size = batch['synth_age'].shape[0]
                
                ## make synth_size as batch size, and we generate each age of synth_age
                image = batch['image'].to(args.device).repeat(synth_size,1,1,1,1)
                sex_tensor = batch['sex'].to(args.device).repeat(synth_size,1)
                modality_tensor = batch['modality'].to(args.device).repeat(synth_size,1)
                another_modality_tensor = 1- modality_tensor
                age_tensor = torch.tensor(batch['synth_age']).to(args.device) / 100
                
                subject_id = batch['image_path'][0].replace(args.input_dir,'.').replace('.nii.gz','')
                save_path = os.path.join(args.output_dir,subject_id)
                if os.path.exists(save_path):                
                    if len(os.listdir(save_path)) == 16:
                        continue
                    else:
                        pass
                latent_output1 = inference_sample(sex_tensor,
                                        another_modality_tensor, ## modality tensor is about target modality
                                        age_tensor,
                                        controlnet,
                                        unet,
                                        image,
                                        noise_scheduler,
                                        inference_step=300,
                                        device = args.device)
                
                output1 = decode(latent_output1,autoencoder)    
                latent_output2 = inference_sample(sex_tensor,
                                        modality_tensor,
                                        age_tensor,
                                        controlnet,
                                        unet,
                                        output1,
                                        noise_scheduler,
                                        inference_step=300,
                                        device = args.device)
                
                output2 = decode(latent_output2,autoencoder)
                ## we trained model to transfer t1 to t2 and t2 to t1 basically
                
                if os.path.exists(save_path) == False:
                    os.makedirs(save_path)
                if batch['modality_str'][0] == 't1': 
                    save_by_synthesis_age(output1,output2,save_path,batch['synth_age'],args)
                elif batch['modality_str'][0] == 't2':
                    save_by_synthesis_age(output2,output1,save_path,batch['synth_age'],args)

        return args
    
    
