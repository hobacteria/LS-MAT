#!/bin/bash
root_dir=./
idBIDS=$1 # sub number
subject_dir=$2 # data dir

SUBJECTS_DIR="${subject_dir}/${idBIDS}"
dir_freesurfer="${subject_dir}/${idBIDS}"
microImage="${SUBJECTS_DIR}/myelin.nii.gz"

outDir="${dir_freesurfer}/anat/surfaces/micro_profiles"
[[ ! -d "$outDir" ]] && mkdir -p "$outDir"

num_surfs=14
tmp="/tmp/${RANDOM}_micapipe_post-MPC_${idBIDS}"
mkdir -p "$tmp"

# Register T1w/T2w intensity to surface
for hemi in lh rh ; do
    unset LD_LIBRARY_PATH
    tot_surfs=$((num_surfs + 2))
    python $root_dir/MPC/generate_equivolumetric_surfaces.py \
            "${dir_freesurfer}/surf/${hemi}.pial" \
            "${dir_freesurfer}/surf/${hemi}.white" \
            "$tot_surfs" \
            "${outDir}/${hemi}.${num_surfs}surfs" \
            "$tmp" \
            --software freesurfer --subject_id "$idBIDS"
    # remove top and bottom surface
    rm -rfv "${outDir}/${hemi}.${num_surfs}surfs0.0.pial" "${outDir}/${hemi}.${num_surfs}surfs1.0.pial"

    # find all equivolumetric surfaces and list by creation time
    x=$(ls -t "$outDir"/"$hemi".${num_surfs}surfs*)
    for n in $(seq 1 1 "$num_surfs") ; do
        which_surf=$(sed -n "$n"p <<< "$x")
	cp "$which_surf" "${dir_freesurfer}/surf/${hemi}.${n}by${num_surf}surf"
        # sample intensity
        ## regheader is subject name, here it is "." while it cause error, ./idBIDS/idBIDS/surf/
        mri_vol2surf \
            --mov "$microImage" \
            --regheader "." \
            --hemi "$hemi" \
            --out_type mgh \
            --interp trilinear \
            --out "${outDir}/${idBIDS}_space-fsnative_desc-${hemi}_MPC-${n}.mgh" \
            --surf "${n}by${num_surf}surf"

        #Remove surfaces used by vol2surf
        rm -rfv "$which_surf" "${dir_freesurfer}/surf/${hemi}.${n}by${num_surf}surf"
        if [[ -f "${outDir}/${idBIDS}_space-fsnative_desc-${hemi}_MPC-${n}.mgh" ]]; then ((Nsteps++)); fi
    done
done

# Register to fsa5/
cp -r /usr/local/freesurfer7/subjects/fsaverage5 ${SUBJECTS_DIR}/
for hemi in lh rh; do
    for n in $(seq 1 1 "$num_surfs"); do
        MPC_fs5="${outDir}/${idBIDS}_space-fsaverage5_desc-${hemi}_MPC-${n}.mgh"

        mri_surf2surf --hemi "$hemi" \
            --srcsubject "." \
            --srcsurfval "${outDir}/${idBIDS}_space-fsnative_desc-${hemi}_MPC-${n}.mgh" \
            --trgsubject fsaverage5 \
            --trgsurfval "$MPC_fs5"
    done
done

# making 18 atlas
atlas_list=$(<$root_dir/MPC/parcellations/atlas_list.txt)
for atlas in $atlas_list
do
	python $root_dir/MPC/surf2mpc.py "$outDir" "$idBIDS" "SINGLE" 14 $atlas "$dir_freesurfer" "$root_dir/MPC"
done

rm -rf ${SUBJECTS_DIR}/fsaverage5
rm -rf $tmp