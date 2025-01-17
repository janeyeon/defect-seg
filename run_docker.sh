docker run --gpus all \
  -v /home:/home \
  -v /data:/data \
  -v /home2:/home2 \
  --name mount_torch2.1.0_cuda11.8 \
  -it mount_torch2.1.0_cuda11.8
