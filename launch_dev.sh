docker run -it -d \
--gpus all \
--privileged \
--network=host \
-e KAGGLE_USERNAME=truongnhathao \
-e KAGGLE_KEY=27dfb69f72f2ac7704f0368eac30fe40 \
-v D:\Projects\Deep_Grasp\deep_grasp_vgu\deep_grasp_vgu:/home/robotlab/deep_grasp_ws/src/deep_grasp_vgu \
haotruongnhat/deep_grasp_vgu /bin/bash