# Use NVIDIA's CUDA base image with Ubuntu 20.04
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set environment variables to prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

# Install necessary packages
RUN apt-get update && \
    apt-get install -y \
        python3.8 \
        python3-pip \
        git \
        wget \
        curl \
        ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a symbolic link for python3
RUN ln -s /usr/bin/python3.8 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch with CUDA support
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install additional Python packages required for Stable Diffusion
RUN pip3 install \
        transformers \
        diffusers \
        scipy \
        ftfy \
        accelerate \
        gradio

# Set the working directory
WORKDIR /app

# Copy your Stable Diffusion scripts into the container
COPY . /app

# Expose the port if your application uses a web interface
EXPOSE 7860

# Set the entry point to your main Python script
CMD ["python", "./src/main.py"]
