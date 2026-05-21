#!/usr/bin/env bash
# run.sh — Launch the LS-MAT pipeline inside Docker
#
# Directory layout expected under DATA_DIR:
#   DATA_DIR/
#     subjects/          raw input images + subjects_age.txt
#     subjects_reg/      created automatically if registration=true
#     output/            synthesized images (created automatically)
#     surfaces/          FastSurfer output (created automatically)
#     output_MPC/        MPC output (created automatically)
#
# Required environment variables (or edit defaults below):
#   DATA_DIR           local (non-NFS/CIFS) directory with subjects/
#   MODELS_DIR         directory containing *.pt checkpoint files
#   FREESURFER_HOME    host path to FreeSurfer (e.g. /usr/local/freesurfer)
#   FSLDIR             host path to FSL (e.g. /usr/local/fsl)
#   FS_LICENSE         path to FreeSurfer license.txt
#   STAGING_DIR        scratch space for FastSurfer I/O — MUST be a local
#                      path mounted at the IDENTICAL path inside the container
#
# Example:
#   DATA_DIR=/home/user/lsmat_data \
#   MODELS_DIR=/home/user/models \
#   FREESURFER_HOME=/usr/local/freesurfer \
#   FSLDIR=/usr/local/fsl \
#   FS_LICENSE=/home/user/freesurfer_license.txt \
#   ./run.sh

set -euo pipefail

DATA_DIR="${DATA_DIR:-/home/$(whoami)/lsmat_data}"
MODELS_DIR="${MODELS_DIR:-${DATA_DIR}/models}"
FREESURFER_HOME="${FREESURFER_HOME:-/usr/local/freesurfer}"
FSLDIR="${FSLDIR:-/usr/local/fsl}"
FS_LICENSE="${FS_LICENSE:-/home/$(whoami)/freesurfer_license.txt}"
STAGING_DIR="${STAGING_DIR:-/tmp/lsmat_staging}"
IMAGE="${IMAGE:-lsmat:latest}"

for var in DATA_DIR MODELS_DIR FREESURFER_HOME FSLDIR FS_LICENSE; do
    val="${!var}"
    if [[ ! -e "$val" ]]; then
        echo "ERROR: $var='$val' does not exist." >&2
        exit 1
    fi
done

mkdir -p "${STAGING_DIR}"
mkdir -p "${DATA_DIR}/output" "${DATA_DIR}/surfaces" "${DATA_DIR}/output_MPC"

# KEY CONSTRAINT: STAGING_DIR must be mounted at the IDENTICAL path inside the
# container because FastSurfer sibling containers are created by the HOST Docker
# daemon, which resolves volume paths on the host filesystem.
docker run --rm --runtime=nvidia \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v "${STAGING_DIR}:${STAGING_DIR}" \
    -v "${DATA_DIR}:/data" \
    -v "${MODELS_DIR}:/models" \
    -v "${FREESURFER_HOME}:/opt/freesurfer:ro" \
    -v "${FSLDIR}:/opt/fsl:ro" \
    -v "${FS_LICENSE}:/fs_license/license.txt:ro" \
    -e LSMAT_STAGING_DIR="${STAGING_DIR}" \
    -e NVIDIA_VISIBLE_DEVICES=all \
    "$IMAGE" \
    --staging-dir "${STAGING_DIR}" \
    "$@"
