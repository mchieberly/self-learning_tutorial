e an image with CUDA and Ubuntu (for GPU support, modify accordingly)
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04

# Install basic utilities
RUN apt-get update && apt-get install -y wget git vim libglib2.0-0 libgl1

# Download and install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Set the path to conda
ENV PATH /opt/miniconda/bin:$PATH

# Set working directory
WORKDIR /workspace

COPY . /opt/mimicdl

# Run the setup script
RUN /bin/bash -c " \
	conda init bash && \
	source ~/.bashrc && \
	conda env create -n mimicdl"

# Keep the container running
CMD ["tail", "-f", "/dev/null"]

# When ready to set up RunPod, from '~/dev/self-learning_tutorial', run:
# `docker build -t mchieberly/mimicdl:latest .`
# `docker push mchieberly/mimicdl:latest`

# When in the container, run:
# `cd /opt/mimicdl`
# `conda activate mimicdl`
# `pip install opencv-python`

# pip install -r requirements
# reinstall torch modules?
# add github ssh key
# pull latest git changes

