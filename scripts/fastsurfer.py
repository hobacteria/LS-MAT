import os
import multiprocessing
import numpy as np
import argparse
import pandas as pd
import os


### 생성된 T1의 segment (GPU)
                  
def process_segment(file_list_path,root_folder,output_folder,parallel_n):
    with open(file_list_path,'r') as f:
        lines = f.readlines()
        n = len(lines)
        print(f'total subject is {n}, split by {parallel_n}')
        
        splited_subjects_array = np.array_split(lines,n//parallel_n + 1)
        
        for splited_subjects in splited_subjects_array:
            with open(f'{root_folder}/tmp.txt','w') as t: ## for parallel processing while considering memory
                for s in splited_subjects:
                        t.write(s) 
            ## in command, we use --seg_only option for segment only
            ## subject_list is tmp.txt, which is splited subjects by parallel_n
            
            command = f'docker run --gpus device=all -v {root_folder}:/data \
                                -v {output_folder}:/output \
                                -v /usr/local/freesurfer:/fs_license \
                                --entrypoint "/fastsurfer/brun_fastsurfer.sh" \
                                --user $(id -u):$(id -g) deepmi/fastsurfer:latest \
                                --fs_license /fs_license/license.txt \
                                --sd /output --subject_list /data/tmp.txt \
                                --parallel --parallel_subjects --3T --seg_only --no_hypothal'

            os.system(command)     
            
def process_surface(file_list_path,root_folder,output_folder,parallel_n):
    with open(file_list_path,'r') as f:
        lines = f.readlines()
        n = len(lines)
        print(f'total subject is {n}, split by {parallel_n}')
        splited_subjects_array = np.array_split(lines,n//parallel_n + 1)
        for splited_subjects in splited_subjects_array:
            incomplete = 0
            
            with open(f'{root_folder}/tmp.txt','w') as t: ## for parallel processing while considering memory
                for s in splited_subjects:
                    if not os.listdir(os.path.join(output_folder,s.split('=')[0])) == ['mri','scripts','stats']:
                        incomplete += 1
                        t.write(s) 
                if incomplete != 0:
                    continue
                   
            command = f'docker run --gpus all -v {root_folder}:/data \
                                -v {output_folder}:/output \
                                -v /usr/local/freesurfer:/fs_license \
                                --entrypoint "/fastsurfer/brun_fastsurfer.sh" \
                                --user $(id -u):$(id -g) deepmi/fastsurfer:latest \
                                --fs_license /fs_license/license.txt \
                                --sd /output --subject_list /data/tmp.txt \
                                --parallel --3T --allow_root --parallel_subjects --surf_only --threads 1'

            os.system(command)
            
## ex) process_segment('/camin1/hkkim/control-net/synthesis_age_pair/T1_subject_list_pair.txt','/camin1/hkkim/control-net/synthesis_age_pair',8)
def fastsurfer(args):
    
    ## make subject list for fastsurfer
    if os.path.exists(args.fastsurfer_dir_path) == False:
        os.makedirs(args.fastsurfer_dir_path)
    surface_folder = os.path.abspath(args.fastsurfer_dir_path)
    """
    you should prepare subjects.txt file in input_dir
    subjects.txt file should have below format
    HCP/749058/T2w_restore_brain/T1w_age_50.0=./output/HCP/749058/T2w_restore_brain/T1w_age_50.0.nii.gz
    This is sample code for out experiment
    """
    
    ## we only need t1 image for fastsurfer
    
    with open(f'{surface_folder}/full_subjects.txt','w') as f: ## subjects.txt is for fastsurfer
        for root, dirs, files in os.walk(args.output_dir):
                for file in files:
                    if ('T1w_age' in file) & ('.nii.gz' in file):
                        full_path = os.path.join(root,file)
                        ## fastsurfer path for docker
                        fastsurfer_path = os.path.join(full_path.replace(args.output_dir,'/data'))
                        
                        subject_id = os.path.join(os.path.dirname(full_path.replace(args.output_dir,'')),file.replace('.nii.gz',''))
                        subject_id =subject_id.replace('/','_')
                        ## fastsurfer format
                        f.write(f'{subject_id}={fastsurfer_path}\n')
    
    ## Because of the computational cost, we use 2 years interval sample for surface analysis. 
    ## read demographic information and sample 2 years interval subjects
    """
    subjects = pd.read_csv(f'{args.input_dir}/subjects.txt',header=None)

    ages = dict()
    for i in range(len(subjects)):
        subject_id = subjects.iloc[i,0].split('/')[2]
        ages[subjects.iloc[i,0]] = subjects.iloc[i,3]
    age_sampling = dict()
    for a in range(10,82,2):
        min_diff= 9999999999
        for k in ages:
            diff=abs(a - ages[k])
            if diff < min_diff:
                age_sampling[a] = k
                min_diff = diff

    sample_subject_id = set([subject_path.split('/')[3] for subject_path in age_sampling.values()])
    
    with open(f'{surface_folder}/subjects.txt','w') as f:
        with open(f'{surface_folder}/full_subjects.txt','r') as t:
            lines = t.readlines()
            for line in lines:
                subject_id = line.split('/')[3]
                if subject_id in sample_subject_id:
                    f.write(line)
    """
            
    
    process_segment(f'{surface_folder}/subjects.txt',os.path.abspath(args.output_dir),surface_folder,8) ## gpu
    process_surface(f'{surface_folder}/subjects.txt',os.path.abspath(args.output_dir),surface_folder,80) ## cpu