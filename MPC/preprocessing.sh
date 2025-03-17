#!/bin/bash

root_dir=./

surface_dir=$1 ## T1
output_dir=$2
threads=$3
task=$4

sub_list=$(<$root_dir/MPC/sub_list_complete.txt)
## original data is freesurfered?
## generated t2 is MNI registered?
## t2_nor is contained in original data?
## 462139
## bash preprocessing.sh /camin1/hkkim/control-net/synthesis /camin1/hkkim/life_gan_restore_brain /camin1/hkkim/control-net/brain_feature 4 fs

if [ $task == fs ]; then
	if command -v parallel > /dev/null 2>&1; then
		# raw data
		parallel --colsep '=' -j $threads mri_convert $surface_dir/{1}/mri/orig.mgz $surface_dir/{1}/T1w_fsnative_brain.nii.gz --out_orientation RAS ::: $sub_list
		

		# bias-field correction & brain extraction

	else
		for sub_line in $sub_list
		do	
			IFS='=' read -r sub t2_path <<< "$sub_line"
			t2_path = "${t2_path//data/}"
			mri_convert $surface_dir/$sub/mri/orig.mgz $surface_dir/$sub/T1w_fsnative_brain.nii.gz		
		done
	fi
fi

if [ $task == myelin ]; then
	for sub_line in $sub_list
	do
		
		IFS=$'=' read -r sub t2_path <<< "$sub_line"
		t2_path=${t2_path//data/}
		t2_path=${t2_path//T1w_age_/T2w_age_}
		# root_dir을 제외한 상대 경로 만들기

		cp -v $output_dir/$t2_path $surface_dir/$sub/T2w_fsnative_brain.nii.gz
		
		python $root_dir/MPC/transform.py --input $surface_dir/$sub/T2w_fsnative_brain.nii.gz --output $surface_dir/$sub/T2w_fsnative_brain.nii.gz 

		## intensity min-max for T1w
	        max_val=$(fslstats $surface_dir/$sub/T1w_fsnative_brain.nii.gz -R | awk '{print $2}')
        	min_val=$(fslstats $surface_dir/$sub/T1w_fsnative_brain.nii.gz -R | awk '{print $1}')
	        fslmaths $surface_dir/$sub/T1w_fsnative_brain.nii.gz -sub $min_val -div $(bc <<< "$max_val - $min_val") $surface_dir/$sub/T1w_nor.nii.gz

		## intensity min-max for T2w
	        max_val=$(fslstats $surface_dir/$sub/T2w_fsnative_brain.nii.gz -R | awk '{print $2}')
        	min_val=$(fslstats $surface_dir/$sub/T2w_fsnative_brain.nii.gz -R | awk '{print $1}')
	        fslmaths $surface_dir/$sub/T2w_fsnative_brain.nii.gz -sub $min_val -div $(bc <<< "$max_val - $min_val") $surface_dir/$sub/T2w_nor.nii.gz

		## calculate T1w/T2w ratio
		wb_command -volume-math "clamp((T1w / T2w), 0, 100)" $surface_dir/$sub/myelin.nii.gz -var T1w $surface_dir/$sub/T1w_nor.nii.gz -var T2w $surface_dir/$sub/T2w_nor.nii.gz -fixnan 0

		## copy output directory
		mkdir -p $surface_dir/$sub
		cp -v $surface_dir/$sub/myelin.nii.gz $surface_dir/$sub/myelin.nii.gz
	done
fi

if [ $task == MPC ]; then
	#if command -v parallel > /dev/null 2>&1; then
	#	parallel --colsep '=' -j $threads "$root_dir/MPC/MPC.sh" {1} "$surface_dir" ::: $sub_list
	#else
	for sub_line in $sub_list
	do		
			IFS='=' read -r sub t2_path <<< "$sub_line"
			source $root_dir/MPC/MPC.sh $sub $surface_dir
	done
	#fi
fi