docker run -it --rm --runtime=nvidia \
--gpus all \
--privileged \
--network=host \
-v $PWD:/home/robotlab/deep_grasp_vgu \
haotruongnhat/deep_grasp_vgu /bin/bash