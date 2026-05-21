# LS-MAT pipeline image
# Baked in: ANTs, Connectome Workbench, Docker CLI, Python packages
# Mounted from host at runtime: FreeSurfer, FSL, data, models (see run.sh)
#
# FastSurfer runs as a sibling Docker container via the host socket.
# The staging path (--staging-dir) must be a HOST-local path mounted
# at the identical path inside this container. See run.sh.

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# ── System packages ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        tcsh bc perl \
        libgomp1 libgl1 libglu1-mesa libxrender1 libxt6 libxmu6 libxss1 \
        libopenblas-dev libpng-dev libjpeg-dev \
        git wget curl ca-certificates unzip \
        parallel \
    && rm -rf /var/lib/apt/lists/*

# ── Docker CLI (for FastSurfer sibling containers) ────────────────────────────
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] \
        https://download.docker.com/linux/ubuntu jammy stable" \
        > /etc/apt/sources.list.d/docker.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends docker-ce-cli \
    && rm -rf /var/lib/apt/lists/*

# ── ANTs 2.5.4 ────────────────────────────────────────────────────────────────
RUN wget -q \
        https://github.com/ANTsX/ANTs/releases/download/v2.5.4/ants-2.5.4-ubuntu-22.04-X64-gcc.zip \
    && unzip -q ants-2.5.4-ubuntu-22.04-X64-gcc.zip \
    && mv ants-2.5.4 /opt/ants \
    && rm ants-2.5.4-ubuntu-22.04-X64-gcc.zip

ENV ANTSPATH=/opt/ants/bin \
    PATH=/opt/ants/bin:${PATH}

# ── Connectome Workbench 2.0.1 ────────────────────────────────────────────────
RUN wget -q \
        "https://www.humanconnectome.org/storage/app/media/workbench/workbench-linux64-v2.0.1.zip" \
    && unzip -q workbench-linux64-v2.0.1.zip -d /usr/local/ \
    && rm workbench-linux64-v2.0.1.zip

ENV PATH=/usr/local/workbench/bin_linux64:${PATH}

# ── FreeSurfer — mounted from host at /opt/freesurfer ────────────────────────
ENV FREESURFER_HOME=/opt/freesurfer \
    FREESURFER=/opt/freesurfer \
    FS_OVERRIDE=1 \
    MNI_DATAPATH=/opt/freesurfer/mni/data \
    PATH=/opt/freesurfer/bin:${PATH}

# ── FSL — mounted from host at /opt/fsl ──────────────────────────────────────
ENV FSLDIR=/opt/fsl \
    FSLOUTPUTTYPE=NIFTI_GZ \
    PATH=/opt/fsl/bin:${PATH}

# ── Python packages ───────────────────────────────────────────────────────────
RUN pip3 install --no-cache-dir --upgrade pip
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# ── LS-MAT source ─────────────────────────────────────────────────────────────
WORKDIR /workspace
COPY . /workspace/

ENTRYPOINT ["python3", "main.py", "--config", "docker-config.json"]
