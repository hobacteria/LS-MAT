import os
import multiprocessing
import numpy as np
import argparse
import pandas as pd

import os
import multiprocessing
import numpy as np


def _check_complete_subject(subject_folder):
    if not os.path.exists(os.path.join(subject_folder,'/label')):
        n = []
    
    else: 
        n = os.listdir(os.path.join(subject_folder,'/label'))
    
    if len(n) != 91:
        return False
    else: 
        return True
    
def run_segment_batch(subject_lines, root_folder, output_folder, index):
    for k, subject_line in enumerate(subject_lines):
        tmp_list_path = f'{root_folder}/tmp_segment_{index}_{k}.txt'
        with open(tmp_list_path, 'w') as f:
            f.write(subject_line)

        subject_nm = subject_line.split('=')[0].split('/')[-1]
        command = f'docker run --gpus device=all -v {root_folder}:/data \
                                -v {output_folder}:/output \
                                -v /usr/local/freesurfer:/fs_license \
                                --entrypoint "/fastsurfer/brun_fastsurfer.sh" \
                                --user $(id -u):$(id -g) deepmi/fastsurfer:latest \
                                --fs_license /fs_license/license.txt \
                                --sd /output --subject_list /data/tmp_segment_{index}_{k}.txt \
                                --parallel --3T --seg_only --no_cereb --no_hypothal --no_biasfield'

        os.system(command)
        os.system(f'cp {output_folder}/{subject_nm}/mri/orig.mgz {output_folder}/{subject_nm}/mri/orig_nu.mgz')
        


def run_surface_batch(subject_lines, root_folder, output_folder, index):
    for k, subject_line in enumerate(subject_lines):
        tmp_list_path = f'{root_folder}/tmp_surface_{index}_{k}.txt'
        with open(tmp_list_path, 'w') as f:
            f.write(subject_line)

        command = f'docker run --gpus all -v {root_folder}:/data \
                                -v {output_folder}:/output \
                                -v /usr/local/freesurfer:/fs_license \
                                --entrypoint "/fastsurfer/brun_fastsurfer.sh" \
                                --user $(id -u):$(id -g) deepmi/fastsurfer:latest \
                                --fs_license /fs_license/license.txt \
                                --sd /output --subject_list /data/tmp_surface_{index}_{k}.txt \
                                --3T --allow_root --surf_only --threads 1'

        os.system(command)


def process_subjects(file_list_path, root_folder, output_folder, parallel_n, task_fn):
    with open(file_list_path, 'r') as f:
        lines = f.readlines()

    n = len(lines)
    print(f'Total subjects: {n}, processing with {parallel_n} parallel workers')

    chunks = np.array_split(lines, parallel_n)

    processes = []
    for i, chunk in enumerate(chunks):
        p = multiprocessing.Process(target=task_fn, args=(chunk, root_folder, output_folder, i))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def process_segment(file_list_path, root_folder, output_folder, parallel_n):
    process_subjects(file_list_path, root_folder, output_folder, parallel_n, run_segment_batch)

def process_surface(file_list_path, root_folder, output_folder, parallel_n):
    process_subjects(file_list_path, root_folder, output_folder, parallel_n, run_surface_batch)

## ex) process_segment('/camin1/hkkim/control-net/synthesis_age_pair/T1_subject_list_pair.txt','/camin1/hkkim/control-net/synthesis_age_pair',8)
def fastsurfer(args):
    
    ## make subject list for fastsurfer
    if os.path.exists(args.fastsurfer_dir_path) == False:
        os.makedirs(args.fastsurfer_dir_path)
    surface_folder = os.path.abspath(args.fastsurfer_dir_path)
    """
    you should prepare subjects.txt file in fastsurfer_dir_path.
    subjects.txt file should have below format.
    subject_id=./output/HCP/749058/T2w_restore_brain/T1w_age_50.0.nii.gz ## subject_id for save directory and path for image
    This is sample code for out experiment
    """
    
    ## we only need t1 image for fastsurfer
    
    with open(f'{surface_folder}/subjects.txt','w') as f: ## subjects.txt is for fastsurfer
        for root, dirs, files in os.walk(args.output_dir):
                for file in files:
                    if ('T1w_age' in file) & ('.nii.gz' in file):
                        full_path = os.path.join(root,file)
                        ## fastsurfer path for docker
                        fastsurfer_path = os.path.join(full_path.replace(args.output_dir,'/data'))
                        
                        subject_id = os.path.join(os.path.dirname(full_path.replace(args.output_dir,'')),file.replace('.nii.gz',''))
                        subject_id =subject_id.replace('/','_')
                        if not _check_complete_subject(os.path.join(surface_folder,subject_id)):
                            os.system(f'rm -rf {os.path.join(args.fastsurfer_dir_path,subject_id)}')
                            ## fastsurfer format
                            f.write(f'{subject_id}={fastsurfer_path}\n')
    
    ## Because of the computational cost, we use 2 years interval sample for surface analysis. 
    ## read demographic information and sample 2 years interval subjects
                
                
    ## process segment and surface    
    process_segment(f'{surface_folder}/subjects.txt',os.path.abspath(args.output_dir),surface_folder,args.n_gpu_parallel) ## gpu
    process_surface(f'{surface_folder}/subjects.txt',os.path.abspath(args.output_dir),surface_folder,args.n_threads) ## cpu
    