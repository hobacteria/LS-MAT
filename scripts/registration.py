import torch
from monai.transforms.transform import MapTransform
from scripts.utils import parse_train_files,dataloader
from tqdm import tqdm
import os


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
        
    ) -> None:
        super().__init__(keys)
        self.device = device 


    def __call__(self, d,save_tmpdir = True):
        """
        This transform can apply on a subject that already registered to the template image by Registeration_image transform.
        
        Raises:
            KeyError: When not ``self.overwriting`` and key already exists in ``data``.

        """
        #image = '/camin2/Database/eNKI/DATA_longitudinal/Surface/V2'
        image = d["image"]
        save_image = d["image_save"]
        
        subject_id = d["subject_id"]
        tmpDir = f'./.tmp/{subject_id}'
        ## you should save image before apply this transform
        command_0 = f'cp {save_image} {tmpDir}/T1w_all_fast_firstseg.nii.gz'
                
        command_7 = f'antsApplyTransforms --default-value 0 -e 3 --input {tmpDir}/T1w_all_fast_firstseg.nii.gz -r {tmpDir}/T1w_restore_brain.nii.gz -o {tmpDir}/T1w_all_fast_seg2b0.nii.gz -t [{tmpDir}/_T1w2b0.mat,1] --interpolation NearestNeighbor'
        
        commands = '> /dev/null \n'.join([command_0,command_7])
        os.system(commands)
        d["inverse_image"] = f'{tmpDir}/T1w_all_fast_seg2b0.nii.gz'
        
        if save_tmpdir:
            #os.system(f'find {tmpDir} -type f ! -name \"T1w_all_fast_seg2b0.nii.gz\" -delete')
            pass
        else:
            os.system(f'find {tmpDir} -type f ! -name \"T1w_all_fast_seg2b0.nii.gz\" -delete')
        
        return d
    
    
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
    
    files = parse_train_files(file = args.input_dir + '/subjects.txt')
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

    with open(args.input_dir + '/subjects.txt', 'r') as f:
        lines = f.readlines()
    with open(args.input_dir + '_reg/subjects.txt', 'w') as f:
        for line in lines:
            line = line.strip()
            train_path,modality,sex,age = line.split(',')
            f.write(f'{train_path}_reg,{modality},sex,age\n')
    args.input_dir = input_regi
    
    return None
