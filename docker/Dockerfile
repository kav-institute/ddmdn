# README!

# Define base image
FROM nvcr.io/nvidia/pytorch:24.10-py3
RUN echo 'root:root' | chpasswd

# Additional apt installations
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y acl sudo build-essential ca-certificates wget openssh-server htop nload python3-dev python3-numpy python3-pip libgl1-mesa-glx \
libavcodec-dev libavformat-dev libswscale-dev libglib2.0-0 libgtk2.0-dev libgtk-3-dev pkg-config ffmpeg python3-tk tk && rm -rf /var/lib/apt/lists/*

# Set ssh daemon to allow root login
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
SHELL ["/bin/bash", "--login", "-c"]

# The place for your repositories and data
RUN mkdir -p /workspace/repos && mkdir /workspace/data

# Set workspace path
WORKDIR /workspace/repos

# Copy entrypoint script and additional python packages
COPY entrypoint.sh /workspace/entrypoint.sh
COPY requirements.txt /workspace/requirements.txt
RUN chmod +x /workspace/entrypoint.sh
RUN python3 -m pip install --upgrade pip
RUN pip install -r /workspace/requirements.txt