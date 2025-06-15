
## MS-Multi-Spine Late Fusion

Submission for the MS-Multi-Spine challenge based on late fusion of predictions from a publicly available segmentation model.


The NVIDIA Container Toolkit needs to be installed to run this image with a GPU. See the installation guide here:
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

To run the container directly with Docker, the device must be specified when using the `docker run` command.
To test that the GPUs are available, run the following command:
```bash
docker run --rm --runtime=nvidia \
          --device=/dev/nvidia0 \
          --device=/dev/nvidiactl \
          --device=/dev/nvidia-uvm \
          --device=/dev/nvidia-uvm-tools \
           docker.io/nvidia/cuda:12.1.0-runtime-ubuntu22.04 \
           nvidia-smi
```

This should print the NVIDIA driver version and the GPUs available on the system.

To run the container with the model, the GPU should have at least 7GB of memory. The first GPU (nvidia0) will be used.
