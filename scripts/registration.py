import torch
import monai
from monai.transforms.transform import MapTransform
from monai.transforms import Compose
from scripts.utils import parse_train_files,dataloader
from tqdm import tqdm
import nibabel as nib
import os
import numpy as np
class Registeration_imaged(MapTransform):

    def __init__(
        self,
        keys,
        templete,
        registraion_method = 'rigid',
        interpolator = 'linear',
        device: torch.device = torch.device("cuda:0"),
        allow_missing_keys: bool = False,
        
    ) -> None:
        super().__init__(keys)
        self.templete = templete
        self.registraion_method = registraion_method
        self.interpolator = interpolator
        self.device = device 


    def __call__(self, d,save_inverse_matrix = True):
        """
        Raises:
            KeyError: When not ``self.overwriting`` and key already exists in ``data``.

        """
        #original_stdout = sys.stdout  # 기존 stdout 저장
        #sys.stdout = open(os.devnull, 'w')  # 출력을 /dev/null로 리디렉션
        
        #image = '/camin2/Database/eNKI/DATA_longitudinal/Surface/V2'
        image = d["image"]
        subject_id = d["subject_id"]
        tmpDir = f'./.tmp/{subject_id}'
        if not os.path.exists(tmpDir):
            os.makedirs(tmpDir)
        command_0 = f'cp {image} {tmpDir}/T1w_restore_brain.nii.gz; cp {self.templete} {tmpDir}/dwi_b0_brain.nii.gz'
        
        
        command_1 = f'antsRegistrationSyN.sh -d 3  -m {tmpDir}/T1w_restore_brain.nii.gz -f {tmpDir}/dwi_b0_brain.nii.gz -o {tmpDir}/T1w2b0 -t r -n 10'
        #command_2 = f'rm -rf {tmpDir}/T1w2b0*Inverse* \n'
        command_3 = f'mv {tmpDir}/T1w2b0Warped.nii.gz {tmpDir}/T1w2b0.nii.gz'
        command_4 = f'mv {tmpDir}/T1w2b00GenericAffine.mat {tmpDir}/_T1w2b0.mat'
        
        commands = '> /dev/null \n'.join([command_0,command_1,command_3,command_4])
        os.system(commands)
        d["image"] = f'{tmpDir}/T1w2b0.nii.gz'
        if save_inverse_matrix:
            d["inverse_matrix"] = f'{tmpDir}/_T1w2b0.mat'
            #os.system(f'shopt -s extglob;rm -rf {tmpDir}/!(T1w2b0.mat|T1w2b0.nii.gz)')
            
        else:
            os.system(f'rm -rf {tmpDir}')
        #sys.stdout = original_stdout
        return d



class Inverse_imaged(MapTransform):

    def __init__(
        self,
        keys,
        device: torch.device = torch.device("cuda:0"),
        allow_missing_keys: bool = False,
        save_tmpdir = True
        
    ) -> None:
        super().__init__(keys)
        self.device = device 
        self.save_tmpdir = save_tmpdir


    def __call__(self, d):
        """
        This transform can apply on a subject that already registered to the template image by Registeration_image transform.
        
        Raises:
            KeyError: When not ``self.overwriting`` and key already exists in ``data``.

        """
        #image = '/camin2/Database/eNKI/DATA_longitudinal/Surface/V2'
        save_image = d["image_save"]
        
        subject_id = d["subject_id"]
        tmpDir = f'./.tmp/{subject_id}'
        ## you should save image before apply this transform
        command_0 = f'cp {save_image} {tmpDir}/T1w_all_fast_firstseg.nii.gz'
                
        command_7 = f'antsApplyTransforms --default-value 0 -e 3 --input {tmpDir}/T1w_all_fast_firstseg.nii.gz -r {tmpDir}/T1w_restore_brain.nii.gz -o {tmpDir}/T1w_all_fast_seg2b0.nii.gz -t [{tmpDir}/_T1w2b0.mat,1] --interpolation NearestNeighbor'
        
        #commands = '> /dev/null \n'.join([command_0,command_7])
        commands = '\n'.join([command_0,command_7])
        os.system(commands)
        d["inverse_image"] = f'{tmpDir}/T1w_all_fast_seg2b0.nii.gz'
        
        if self.save_tmpdir:
            #os.system(f'find {tmpDir} -type f ! -name \"T1w_all_fast_seg2b0.nii.gz\" -delete')
            pass
        else:
            os.system(f'find {tmpDir} -type f ! -name \"T1w_all_fast_seg2b0.nii.gz\" -delete')
        
        return d
    
class Save_imaged(MapTransform):

    def __init__(
        self,
        keys,
        device: torch.device = torch.device("cuda:0"),
        allow_missing_keys: bool = False,
        
    ) -> None:
        super().__init__(keys)
        self.device = device 


    def __call__(self, d):
        image = d['image_save']
        image_path = d['path_copy']
        subject_id = d['subject_id']
        image = image[0].cpu().numpy().astype(np.float32)
        affine = nib.load(f'./.tmp/{subject_id}/T1w2b0.nii.gz').affine
        
        nib.save(nib.Nifti1Image(image, affine), image_path)
        d['image_save'] = d['path_copy']
        return d

inverse_transforms = Compose(
    [   
        monai.transforms.LoadImaged(keys=["image_save"],allow_missing_keys=False),
        monai.transforms.EnsureChannelFirstd(keys=["image_save"],allow_missing_keys=True),
        monai.transforms.Orientationd(keys=["image_save"], axcodes="RAS"),
        monai.transforms.Resized(keys=["image_save"], spatial_size=(227,272,227), mode="trilinear"),
        Save_imaged(keys = ['image_save']),
        Inverse_imaged(keys = ['image_save'],save_tmpdir=True)
        
        #monai.transforms.Lambdad(
        #    keys="sex", func=lambda x: sex_mapping[x['sex']].to(device)),
        #monai.transforms.Lambdad(
        #    keys="age", func=lambda x: torch.Tensor([x['age']]).type(torch.float).to(device)),
    ]
)



def registration(args):
    """
    This function is for registration of input images.
    If input files are registered, set 0. If not, set 1
    """
    if args.registration == 1:
        # do registration
        print('do registration')
        
    else:
        print('no registration')
        return args
    
    files = parse_train_files(file = args.input_dir + args.subjects_info)
    input_regi = args.input_dir + '_reg'
    if os.path.exists(input_regi) == False:
        os.makedirs(input_regi)
    if os.path.exists('./.tmp') == False:
        os.makedirs('./.tmp')
        
    t1_regi_transform = Registeration_imaged(keys = 'image',templete='./template/MNI152_T1_0.8mm_brain.nii.gz')
    t2_regi_transform = Registeration_imaged(keys = 'image',templete='./template/MNI152_T2_0.8mm_brain.nii.gz')
    
    with tqdm(files) as pbar:
        for image_info in pbar:
            image_path = image_info['image']
            modality = image_info['modality']         
            subject_id = image_path.replace(args.input_dir,'.').replace('.nii.gz','') # define filename and subfolder structure automatically
            image_info['subject_id'] = subject_id
            pbar.set_description(f"Processing {image_path}")
            if modality == 't1':
                image_reg = t1_regi_transform(image_info) ## process registration, image will save in tmp folder
                image_reg_path = image_reg['image']
                if not os.path.exists(os.path.dirname(f'{input_regi}/{subject_id}')):
                    os.makedirs(os.path.dirname(f'{input_regi}/{subject_id}'))
                os.system(f'cp {image_reg_path} {input_regi}/{subject_id}_reg.nii.gz') # copy registered image to input_regi folder
                
            elif modality == 't2':
                image_reg = t2_regi_transform(image_info)
                image_reg_path = image_reg['image']
                if not os.path.exists(os.path.dirname(f'{input_regi}/{subject_id}')):
                    os.makedirs(os.path.dirname(f'{input_regi}/{subject_id}'))
                os.system(f'cp {image_reg_path} {input_regi}/{subject_id}_reg.nii.gz')
            else:
                print(f'Unknown modality: {image_path}')
                pass

    with open(args.input_dir + f'_reg/{args.subjects_info}', 'w') as f:
        for file in files:
            train_path,modality,sex,age,synth_age = file['image_path'],file['modality'],file['sex'],file['age'],file['synth_age']
            train_path_reg = train_path.replace(args.input_dir,input_regi).replace('.nii.gz','_reg.nii.gz')
            f.write(f'{train_path_reg},{modality},{sex},{age},{synth_age}\n')
    args.input_dir = input_regi
    
    return args
