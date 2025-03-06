import os
import pandas as pd

root_dir = './surfaces'
all_subjects = os.listdir(root_dir)
miss_sub = []
anat_subject = []
for s in all_subjects:
    if not os.path.exists(os.path.join(root_dir,s) + '/label'):
        n = []
    
    else: 
        n = os.listdir(os.path.join(root_dir,s) + '/label')
    
    if len(n) != 91:
        miss_sub.append(s)
        print(s + ' is not complete.')
    else: anat_subject.append(s)
        
print('Total number of subjects is ' + str(len(all_subjects)), 'Total number of missing subjects is ' + str(len(miss_sub)))

with open('./sub_list.txt','w') as f:
    for sub in anat_subject:
        f.write(sub + '\n')
"""
#root_path = '/camin1/hkkim/control-net/synthesis_age'
root_path = '/camin1/hkkim/control-net/synthesis_age_pair'
os.system('sudo ls')
for root, dir, files in os.walk(root_path):
    for f in files:
        if 'nii.gz' in f:
            additional_str = f.split('restore_brain_syn')[1].replace('.nii.gz','')
            adj_path = root + additional_str + '/' + f.replace(additional_str,'')
            origianl_path = root + '/' + f
            if not os.path.exists(os.path.dirname(adj_path)):
                os.mkdir(os.path.dirname(adj_path))
            os.system(f'mv {origianl_path} {adj_path}')
        else: continue
"""
    
# case in T1 from brain_feature
#T1_path = '/camin1/hkkim/control-net/brain_feature'
#T2_path = '/camin1/hkkim/control-net/synthesis_age_pair'
# case in T1 from brain_feature_pair
T1_path = ''
T2_path = T1_path.replace('T1w_age','T2w_age')

output_dir = '/camin1/hkkim/control-net/MPC_out'



threads = 1
fs_command = f'bash ./custom/MPC/preprocessing.sh {T1_path} {T2_path} {output_dir} {threads} fs'
myelin_command = f'bash ./custom/MPC/preprocessing.sh {T1_path} {T2_path} {output_dir} {threads} myelin'
MPC_command = f'bash ./custom/MPC/preprocessing.sh {T1_path} {T2_path} {output_dir} {threads} MPC'
os.system(fs_command)
os.system(myelin_command)
os.system(MPC_command)

def excute_MPC_process(T1_path,T2_path,output_dir,n_treads):
    threads = n_treads
    fs_command = f'bash ./custom/MPC/preprocessing.sh {T1_path} {T2_path} {output_dir} {threads} fs'
    myelin_command = f'bash ./custom/MPC/preprocessing.sh {T1_path} {T2_path} {output_dir} {threads} myelin'
    MPC_command = f'bash ./custom/MPC/preprocessing.sh {T1_path} {T2_path} {output_dir} {threads} MPC'
    os.system(fs_command)
    os.system(myelin_command)
    os.system(MPC_command)
    
excute_MPC_process(T1_path,T2_path,output_dir,70)