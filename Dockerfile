### Build ROS-supported environment
FROM nvidia/cuda:11.4.2-devel-ubuntu20.04 as base

ENV ROS_DISTRO=noetic

ENV DEBIAN_FRONTEND=noninteractive 
ENV DEBCONF_NONINTERACTIVE_SEEN=true

RUN apt-get update && apt-get install -y \
      sudo \
      python-dev \
      python3-pip \
      iputils-ping \ 
      netcat \
      libnvidia-gl-440 \
      software-properties-common \
      git \
      tmux \
      && \
    rm -rf /var/lib/apt/lists/*

ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

##### Setup user
RUN useradd -m robotlab && echo "robotlab:robotlab" | chpasswd && adduser robotlab sudo
USER robotlab

WORKDIR /home/robotlab
ENV HOME=/home/robotlab

USER root
### Install ROS
RUN echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros1-latest.list
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN /bin/bash -c "apt update && apt install -y ros-noetic-desktop"

RUN echo 'alias source_ros="source /opt/ros/${ROS_DISTRO}/setup.bash"' >> ~/.bashrc
#RUN echo 'source /opt/ros/${ROS_DISTRO}/setup.bash' >> ~/.bashrc

FROM base AS torch

USER robotlab
CMD /bin/bash

ENV PATH="${HOME}/.local/bin:$PATH"

COPY ./requirements.txt ${HOME}/requirements.txt
RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install -r ${HOME}/requirements.txt
RUN python3 -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

WORKDIR ${HOME}

RUN mkdir -p ${HOME}/deep_grasp_ws/src

WORKDIR ${HOME}/deep_grasp_ws/src
RUN git clone https://github.com/haotruongnhat/deep_grasp_msgs

COPY ./deep_grasp_vgu ${HOME}/deep_grasp_ws/src/deep_grasp_vgu

WORKDIR ${HOME}/deep_grasp_ws
RUN /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash && catkin_make"