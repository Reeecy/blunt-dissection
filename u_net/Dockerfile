FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

# Install OS-level dependencies
RUN apt-get update && apt-get install -y \
    python3-opencv \
    libgl1-mesa-glx \
    git \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Configure timezone automatically
RUN ln -fs /usr/share/zoneinfo/Europe/Budapest /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install \
        albumentations \
        segmentation-models-pytorch \
        timm \
        matplotlib \
        scikit-learn \
        tqdm \
        opencv-python

WORKDIR /workspace

CMD ["/bin/bash"]
