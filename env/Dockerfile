FROM nvidia/cuda:12.0.0-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    apt-utils \
    build-essential \
    git \
    gcc \
    python3.9 \
    python3-dev \
    python3-pip \
    wget \
    libglib2.0-0 \
    libopenexr-dev libxi-dev \
    libglfw3-dev libglew-dev libomp-dev libxinerama-dev libxcursor-dev \
    libgles2-mesa-dev libboost-all-dev libomp-dev python-opengl \
    pkg-config \
    libjpeg-dev libtiff5-dev libpng-dev \
    ffmpeg libavcodec-dev libavformat-dev libswscale-dev libxvidcore-dev libx264-dev libxine2-dev \
    libv4l-dev v4l-utils \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libgtk-3-dev \
    mesa-utils libgl1-mesa-dri libgtkgl2.0-dev libgtkglext1-dev \
    libatlas-base-dev gfortran libeigen3-dev \
    unzip

RUN wget https://cmake.org/files/v3.21/cmake-3.21.7-linux-x86_64.tar.gz
RUN tar xzf cmake-3.21.7-linux-x86_64.tar.gz
RUN rm -rf cmake-3.21.7-linux-x86_64.tar.gz​
RUN cd cmake-3.21.7-linux-x86_64
ENV PATH="/cmake-3.21.7-linux-x86_64/bin:$PATH"
RUN echo $PATH
RUN cmake --version
COPY requirements.txt /working/requirements.txt
RUN pip install -r /working/requirements.txt --default-timeout=3000
RUN pip3 install torch torchvision --default-timeout=3000
