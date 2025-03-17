import os
import pandas as pd



def  MPC_calc(args):

    root_dir = args.fastsurfer_dir_path
    with open(args.fastsurfer_dir_path+'/subjects.txt','r') as f: 
        all_subjects = f.readlines()
        all_subjects = [l.strip('\n') for l in all_subjects]
        
    miss_sub = []
    anat_subject = []
    for sub_line in all_subjects:
        s = sub_line.split('=')[0]
        if not os.path.exists(os.path.join(root_dir,s) + '/label'):
            n = []
        
        else: 
            n = os.listdir(os.path.join(root_dir,s) + '/label')
        
        if len(n) != 91:
            miss_sub.append(s)
            print(s + ' is not complete.')
        else: anat_subject.append(sub_line)
            
    print('Total number of subjects is ' + str(len(all_subjects)), 'Total number of missing subjects is ' + str(len(miss_sub)))

    with open('./MPC/sub_list_complete.txt','w') as f:
        
        for sub in anat_subject:
            f.write(sub+ '\n')

        
    # case in T1 from brain_feature
    #T1_path = '/camin1/hkkim/control-net/brain_feature'
    #T2_path = '/camin1/hkkim/control-net/synthesis_age_pair'
    # case in T1 from brain_feature_pair

    T1_path = args.fastsurfer_dir_path
    output_dir = args.output_dir
    

    threads = 40
    fs_command = f'bash ./MPC/preprocessing.sh {T1_path} {output_dir} {threads} fs'
    myelin_command = f'bash ./MPC/preprocessing.sh {T1_path} {output_dir} {threads} myelin'
    MPC_command = f'bash ./MPC/preprocessing.sh {T1_path} {output_dir} {threads} MPC'
    os.system(fs_command)
    os.system(myelin_command)
    os.system(MPC_command)

    def excute_MPC_process(T1_path,output_dir,n_treads):
        threads = n_treads
        fs_command = f'bash ./MPC/preprocessing.sh {T1_path} {output_dir} {threads} fs'
        myelin_command = f'bash ./MPC/preprocessing.sh {T1_path} {output_dir} {threads} myelin'
        MPC_command = f'bash ./MPC/preprocessing.sh {T1_path} {output_dir} {threads} MPC'
        os.system(fs_command)
        os.system(myelin_command)
        os.system(MPC_command)
        
    excute_MPC_process(T1_path,output_dir,70)