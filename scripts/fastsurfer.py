import os
import shutil
import multiprocessing
import numpy as np


def _seg_complete(subject_folder):
    return os.path.exists(os.path.join(subject_folder, 'mri', 'aparc.DKTatlas+aseg.deep.mgz'))

def _surf_complete(subject_folder):
    label_dir = os.path.join(subject_folder, 'label')
    return os.path.exists(label_dir) and len(os.listdir(label_dir)) == 91


def _stage_image(host_image_path, local_data_dir, docker_rel_path):
    """Copy a single image from CIFS to local staging dir, preserving relative path."""
    dest = os.path.join(local_data_dir, docker_rel_path.lstrip('/'))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not os.path.exists(dest):
        shutil.copy2(host_image_path, dest)
    return dest


def run_segment_batch(subject_lines, output_dir, surface_folder, index, fs_license_path, local_dir):
    """Run FastSurfer segmentation for a chunk of subjects.

    Images are staged to local_dir (accessible by snap Docker) before Docker runs.
    Output is copied back to surface_folder (CIFS) after completion.
    """
    local_data = os.path.join(local_dir, 'data')
    local_output = os.path.join(local_dir, 'output')
    os.makedirs(local_data, exist_ok=True)
    os.makedirs(local_output, exist_ok=True)

    for k, subject_line in enumerate(subject_lines):
        subject_line = subject_line.strip()
        if not subject_line:
            continue
        subject_id, docker_image_path = subject_line.split('=', 1)

        if _seg_complete(os.path.join(surface_folder, subject_id)):
            continue

        # Stage input image from CIFS to local filesystem
        host_image_path = os.path.join(output_dir, docker_image_path.replace('/data/', '', 1))
        _stage_image(host_image_path, local_data, docker_image_path.replace('/data', ''))

        # Write subject list to local data dir
        tmp_list = os.path.join(local_data, f'tmp_segment_{index}_{k}.txt')
        with open(tmp_list, 'w') as f:
            f.write(f'{subject_id}={docker_image_path}\n')

        command = (
            f'docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all'
            f' -v {local_data}:/data'
            f' -v {local_output}:/output'
            f' -v {fs_license_path}:/fs_license/license.txt'
            f' --entrypoint "/fastsurfer/brun_fastsurfer.sh"'
            f' --user $(id -u):$(id -g) deepmi/fastsurfer:latest'
            f' --fs_license /fs_license/license.txt'
            f' --sd /output --subject_list /data/tmp_segment_{index}_{k}.txt'
            f' --parallel --3T --seg_only --no_cereb --no_hypothal --no_biasfield'
        )
        os.system(command)

        # orig_nu.mgz needed for surface step
        orig = os.path.join(local_output, subject_id, 'mri', 'orig.mgz')
        orig_nu = os.path.join(local_output, subject_id, 'mri', 'orig_nu.mgz')
        if os.path.exists(orig) and not os.path.exists(orig_nu):
            shutil.copy2(orig, orig_nu)

        # Copy output back to CIFS surface_folder
        local_subject_out = os.path.join(local_output, subject_id)
        cifs_subject_out = os.path.join(surface_folder, subject_id)
        if os.path.exists(local_subject_out):
            shutil.copytree(local_subject_out, cifs_subject_out, dirs_exist_ok=True)


def run_surface_batch(subject_lines, output_dir, surface_folder, index, fs_license_path, local_dir):
    """Run FastSurfer surface reconstruction for a chunk of subjects.

    Stages previously-completed segmentation output from CIFS to local filesystem.
    """
    local_data = os.path.join(local_dir, 'data')
    local_output = os.path.join(local_dir, 'output')
    os.makedirs(local_data, exist_ok=True)
    os.makedirs(local_output, exist_ok=True)

    for k, subject_line in enumerate(subject_lines):
        subject_line = subject_line.strip()
        if not subject_line:
            continue
        subject_id, docker_image_path = subject_line.split('=', 1)

        if _surf_complete(os.path.join(surface_folder, subject_id)):
            continue

        # Stage mri/ entirely (recon-surf needs norm.mgz, transforms/, etc.)
        # but exclude scripts/ logs — their presence triggers recon-surf's
        # "existing directory" safety check.
        cifs_subject_seg = os.path.join(surface_folder, subject_id)
        local_subject_out = os.path.join(local_output, subject_id)
        if os.path.exists(local_subject_out):
            shutil.rmtree(local_subject_out)
        if os.path.exists(cifs_subject_seg):
            shutil.copytree(
                cifs_subject_seg, local_subject_out,
                ignore=shutil.ignore_patterns('build.log', 'deep-seg.log',
                                              'recon-surf.log'),
            )
        else:
            os.makedirs(os.path.join(local_subject_out, 'mri'), exist_ok=True)

        # Stage input image
        host_image_path = os.path.join(output_dir, docker_image_path.replace('/data/', '', 1))
        _stage_image(host_image_path, local_data, docker_image_path.replace('/data', ''))

        tmp_list = os.path.join(local_data, f'tmp_surface_{index}_{k}.txt')
        with open(tmp_list, 'w') as f:
            f.write(f'{subject_id}={docker_image_path}\n')

        command = (
            f'docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all'
            f' -v {local_data}:/data'
            f' -v {local_output}:/output'
            f' -v {fs_license_path}:/fs_license/license.txt'
            f' --entrypoint "/fastsurfer/brun_fastsurfer.sh"'
            f' --user $(id -u):$(id -g) deepmi/fastsurfer:latest'
            f' --fs_license /fs_license/license.txt'
            f' --sd /output --subject_list /data/tmp_surface_{index}_{k}.txt'
            f' --3T --allow_root --surf_only --threads 1'
        )
        os.system(command)

        # Copy surface output back to CIFS
        if os.path.exists(local_subject_out):
            shutil.copytree(local_subject_out, cifs_subject_seg, dirs_exist_ok=True)


def process_subjects(file_list_path, output_dir, surface_folder, parallel_n, task_fn, fs_license_path, local_dir):
    with open(file_list_path, 'r') as f:
        lines = [l for l in f.readlines() if l.strip()]

    n = len(lines)
    print(f'Total subjects: {n}, processing with {parallel_n} parallel workers')

    chunks = np.array_split(lines, min(parallel_n, n))

    processes = []
    for i, chunk in enumerate(chunks):
        p = multiprocessing.Process(
            target=task_fn,
            args=(list(chunk), output_dir, surface_folder, i, fs_license_path, local_dir)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def process_segment(file_list_path, output_dir, surface_folder, parallel_n, fs_license_path, local_dir):
    process_subjects(file_list_path, output_dir, surface_folder, parallel_n, run_segment_batch, fs_license_path, local_dir)


def process_surface(file_list_path, output_dir, surface_folder, parallel_n, fs_license_path, local_dir):
    process_subjects(file_list_path, output_dir, surface_folder, parallel_n, run_surface_batch, fs_license_path, local_dir)


def fastsurfer(args):
    os.makedirs(args.fastsurfer_dir_path, exist_ok=True)
    surface_folder = os.path.abspath(args.fastsurfer_dir_path)
    output_dir = os.path.abspath(args.output_dir)
    fs_license_path = os.path.abspath(args.fs_license_path)
    local_dir = os.path.abspath(args.fastsurfer_local_dir)
    os.makedirs(local_dir, exist_ok=True)

    subjects_txt = os.path.join(surface_folder, 'subjects.txt')
    with open(subjects_txt, 'w') as f:
        for root, dirs, files in os.walk(output_dir):
            for file in sorted(files):
                if 'T1w_age' in file and file.endswith('.nii.gz'):
                    full_path = os.path.join(root, file)
                    docker_path = '/data' + full_path.replace(output_dir, '')
                    subject_id = os.path.join(
                        os.path.dirname(full_path.replace(output_dir, '')),
                        file.replace('.nii.gz', '')
                    ).replace('/', '_').lstrip('_')
                    f.write(f'{subject_id}={docker_path}\n')

    process_segment(subjects_txt, output_dir, surface_folder, args.n_gpu_parallel, fs_license_path, local_dir)
    process_surface(subjects_txt, output_dir, surface_folder, args.n_threads, fs_license_path, local_dir)
