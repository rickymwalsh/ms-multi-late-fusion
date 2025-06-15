#FROM python:3.11-slim    ubuntu
FROM docker.io/nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install system dependencies, Python3.11 and requirements for SCT
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3.11 python3-pip && \
    apt-get install -y --no-install-recommends git unzip && \
    apt-get install -y --no-install-recommends curl bzip2 libglib2.0-0 gcc && \
    rm -rf /var/lib/apt/lists/*

## Create app directory
#RUN mkdir /app
## Copy program files into the container
#COPY ./inference /app/inference
#COPY ./requirements.txt /app/requirements.txt

# Copy the files from the GitHub repository
RUN git clone https://github.com/rickymwalsh/ms-multi-late-fusion.git app

RUN cd /app && git clone -b rw/tta_pmaps https://github.com/rickymwalsh/spinalcordtoolbox.git sct
RUN cd /app/sct && ./install_sct -iygc
# Set environment variables for SCT
ENV PATH="/app/sct/bin:${PATH}"
# Install required tasks/segmentation models
RUN sct_deepseg lesion_ms -install
RUN sct_deepseg spinalcord -install

# Install app dependencies
RUN cd /app && pip install --no-cache-dir -r requirements.txt

