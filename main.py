import argparse
import os
os.chdir('/camin1/hkkim/controlnet-infer')
from scripts.generate_image import generate_image
from scripts.registration import registration
from scripts.fastsurfer import fastsurfer
from scripts.utils import load_config
#from scripts.gradient import gradient
import pandas as pd
from tqdm import tqdm

"""
with open('./validation_subject.txt','r') as f:
    validation_subjects = f.readlines()
    validation_subjects = [x.split(',')[0] for x in validation_subjects]

validation_subjects = [s.replace('/camin1/hkkim/control-net/embeddings','./subjects').replace('\n','').replace('_emb','') for s in validation_subjects]
subject_info = pd.read_csv('./subjects/all_subject.csv')
for path in validation_subjects:
    root = os.path.dirname(path)
    file = os.path.basename(path)
    
    if 'nii.gz' in file:
        if not os.path.exists(path):
            print(f"{path} does not exist. Please check the input path")
            break
        subject = root.split('/')[-1].split('_')[0]
        src_data = root.split('/')[-2]
        
        info = subject_info[subject_info['src_subject_id'].str.contains(subject)]
        if len(info) == 0:
            print(f'no info for {subject}')
        elif len(info) > 1:
            info = info[info['sub_div'] == src_data]
        with open('./subjects/subjects.txt','a') as f:
            sex = info['sex'].values[0].lower()
            age = info['interview_age'].values[0]
            
            f.write(f"{os.path.join(root,file)},{file[:2].lower()},{sex},{age},[10,20,30,40,50,60,70,80]\n")

"""

## get args with argparse
parser = argparse.ArgumentParser(description="args for inference")
parser.add_argument("-c", "--config", type=str, default="config.json", help="Path to config JSON file")
args = parser.parse_args()
args = load_config(args.config)

if os.path.exists(args.output_dir) == False:
    os.makedirs(args.output_dir)
if os.path.exists(args.input_dir + '/subjects.txt') == False:
        print("subjects.txt file does not exist. Please check the input path")
        exit(0)
## args.registration = 1
## check registration option and do registration if needed
args = registration(args)
## get path and pass to vaegan, save latent space
## get path and pass to control-LDM, save generated latent space & save generated images
generate_image(args)
## extracte features from generated images (fastsurfer)
if args.fastsurfer == 1:
    fastsurfer(args)
## calulate gradient of generated images
if args.gradient == 1:
    gradient(args)