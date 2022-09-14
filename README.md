# deep_grasp_vgu

Docker build based on different pre-built dockers:

- ur3\docker\Dockerfile -> haotruongnhat/ros-ur3:melodic 
- deep_grasp_vgu\Dockerfile.ur3_base -> haotruongnhat/deep_grasp_vgu

Rebuild Moveit for python3: https://moveit.ros.org/install/source/
```
wstool init src
wstool merge -t src https://raw.githubusercontent.com/ros-planning/moveit/master/moveit.rosinstall
wstool update -t src
rosdep install -y --from-paths src --ignore-src --rosdistro ${ROS_DISTRO}
catkin build -DPYTHON_EXECUTABLE=/usr/bin/python3 \
            -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so \
            -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m \
            -DPYTHON_VERSION=3
```


Required for Docker rebuild:

Wheels:
- https://download.pytorch.org/whl/cu111/torch-1.10.0%2Bcu111-cp36-cp36m-linux_x86_64.whl

Clone from:

- https://github.com/dougsm/ggcnn
- https://github.com/dougsm/mvp_grasp

Download datasets and weights:

- `cd datasets && kaggle datasets download -d oneoneliu/cornell-grasp`
- `wget https://github.com/dougsm/ggcnn/releases/download/v0.1/ggcnn_weights_cornell.zip`