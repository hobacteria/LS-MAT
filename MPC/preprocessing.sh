#!/bin/bash

root_dir=/camin1/hkkim/control-net/custom

input_dir=$1 ## T1
raw_path=$2 ## T2
output_dir=$3
threads=$4
task=$5

sub_list=$(<$output_dir/sub_list.txt)
## original data is freesurfered?
## generated t2 is MNI registered?
## t2_nor is contained in original data?
## 462139
## bash preprocessing.sh /camin1/hkkim/control-net/synthesis /camin1/hkkim/life_gan_restore_brain /camin1/hkkim/control-net/brain_feature 4 fs

if [ $task == fs ]; then
	if command -v parallel > /dev/null 2>&1; then
		# raw data
		parallel -j $threads mri_convert $input_dir/{}/mri/orig.mgz $input_dir/{}/T1w_fsnative_brain.nii.gz --out_orientation RAS ::: $sub_list

		# bias-field correction & brain extraction

	else
		for sub in $sub_list
		do
			mri_convert $input_dir/$sub/mri/orig.mgz $input_dir/$sub/T1w_fsnative_brain.nii.gz		
		done
	fi
fi

if [ $task == myelin ]; then
	for sub in $sub_list
	do
		
		found_paths=$(find "$raw_path" -type d -name "$sub" 2>/dev/null)
		# root_dir을 제외한 상대 경로 만들기
		cp -v $found_paths/T2w_restore_brain_syn.nii.gz $input_dir/$sub/T2w_fsnative_brain.nii.gz
		
		python $root_dir/MPC/transform.py --input $input_dir/$sub/T2w_fsnative_brain.nii.gz --output $input_dir/$sub/T2w_fsnative_brain.nii.gz 

		# intensity min-max for T1w
	        max_val=$(fslstats $input_dir/$sub/T1w_fsnative_brain.nii.gz -R | awk '{print $2}')
        	min_val=$(fslstats $input_dir/$sub/T1w_fsnative_brain.nii.gz -R | awk '{print $1}')
	        fslmaths $input_dir/$sub/T1w_fsnative_brain.nii.gz -sub $min_val -div $(bc <<< "$max_val - $min_val") $input_dir/$sub/T1w_nor.nii.gz

		# intensity min-max for T2w
	        max_val=$(fslstats $input_dir/$sub/T2w_fsnative_brain.nii.gz -R | awk '{print $2}')
        	min_val=$(fslstats $input_dir/$sub/T2w_fsnative_brain.nii.gz -R | awk '{print $1}')
	        fslmaths $input_dir/$sub/T2w_fsnative_brain.nii.gz -sub $min_val -div $(bc <<< "$max_val - $min_val") $input_dir/$sub/T2w_nor.nii.gz

		# calculate T1w/T2w ratio
		wb_command -volume-math "clamp((T1w / T2w), 0, 100)" $input_dir/$sub/myelin.nii.gz -var T1w $input_dir/$sub/T1w_nor.nii.gz -var T2w $input_dir/$sub/T2w_nor.nii.gz -fixnan 0

		# copy output directory
		mkdir -p $output_dir/$sub
		cp -v $input_dir/$sub/myelin.nii.gz $output_dir/$sub/myelin.nii.gz
	done
fi

if [ $task == MPC ]; then
	if command -v parallel > /dev/null 2>&1; then
		parallel -j $threads $root_dir/MPC/MPC.sh {} $input_dir ::: $sub_list
	else
		for sub in $sub_list
		do
		        source $root_dir/MPC.sh $sub $input_dir
		done
	fi
fi