### Build ROS-supported environment
FROM nvidia/cuda:11.4.2-devel-ubuntu18.04 as base

ENV ROS_DISTRO=melodic

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

USER root
### Install ROS
RUN echo "deb http://packages.ros.org/ros/ubuntu bionic main" > /etc/apt/sources.list.d/ros1-latest.list
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN /bin/bash -c "apt update && apt install -y ros-melodic-ros-base"

RUN echo 'alias source_ros="source /opt/ros/${ROS_DISTRO}/setup.bash"' >> ~/.bashrc
#RUN echo 'source /opt/ros/${ROS_DISTRO}/setup.bash' >> ~/.bashrc

FROM base AS torch

USER root
RUN apt-get install -y python3-rosdep

ENV ABC=a
COPY ./requirements.txt /requirements.txt
RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install --ignore-installed -r /requirements.txt
RUN python3 -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

WORKDIR /home/robotlab
ENV HOME=/home/robotlab

RUN mkdir -p ${HOME}/deep_grasp_ws/src

WORKDIR ${HOME}/deep_grasp_ws/src
RUN git clone https://github.com/haotruongnhat/deep_grasp_msgs
RUN git clone https://github.com/ros/geometry2 -v 0.6.5

RUN apt-get install -y ros-$ROS_DISTRO-catkin \
                        python-catkin-tools \
                        python3-empy

# Install SIP 4.19.8
ADD ./dependencies/sip-4.19.8.tar.gz $HOME

RUN /bin/bash -c "cd ~/ \
               && cd sip-4.19.8 \
               && python3 configure.py \
               && make -j4 && make install"

# Install PyKDL
ADD ./dependencies/orocos_kinematics_dynamics.tar.xz $HOME
RUN apt -y install libeigen3-dev && rm -rf /var/lib/apt/lists/*
RUN /bin/bash -c "cd ~/orocos_kinematics_dynamics/orocos_kdl \
               && mkdir build && cd build \
               && cmake -DCMAKE_BUILD_TYPE=Release .. \
               && make -j4 && make install"

RUN /bin/bash -c "cd ~/orocos_kinematics_dynamics/python_orocos_kdl \
               && mkdir build && cd build \
               && cmake -DCMAKE_BUILD_TYPE=Release \
               -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m \
               -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so \
               -DPYTHON_VERSION=3 .. \
               && make -j4"


COPY ./deep_grasp_vgu ${HOME}/deep_grasp_ws/src/deep_grasp_vgu

WORKDIR ${HOME}/deep_grasp_ws

# RUN rosdep init
# RUN rosdep update
# RUN rosdep install --from-paths src --ignore-src --rosdistro=$ROS_DISTRO -y

##
RUN apt-get update && apt-get install -y ros-melodic-tf2-bullet


# Compiling ros workspace
RUN /bin/bash -c "source /opt/ros/$ROS_DISTRO/setup.bash \
               && catkin build -DCMAKE_BUILD_TYPE=Release \
                                -DPYTHON_EXECUTABLE=/usr/bin/python3 \
                                -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so \
                                -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m \
                                -DPYTHON_VERSION=3"

RUN echo 'source /opt/ros/$ROS_DISTRO/setup.bash' >> ~/.bashrc
RUN echo 'source ${HOME}/deep_grasp_ws/devel/setup.bash' >> ~/.bashrc

# USER robotlab
# CMD /bin/bash

ENV PATH="${HOME}/.local/bin:$PATH"

