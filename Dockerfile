FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04

# Install packages
RUN apt update
RUN apt -y install libgl1-mesa-glx
RUN apt -y install libglib2.0-dev
RUN apt-get install -y wget git build-essential zip unzip
 

 
# install Miniconda (or Anaconda)
RUN wget https://repo.continuum.io/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh \
    && /bin/bash Miniconda3-py38_4.9.2-Linux-x86_64.sh	 -b -p /softwares/miniconda3 \
    && rm -v Miniconda3-py38_4.9.2-Linux-x86_64.sh
ENV PATH "/softwares/miniconda3/bin:${PATH}"
ENV LD_LIBRARY_PATH="/softwares/miniconda3/lib:${LD_LIBRARY_PATH}"
 
 # install Python packages
RUN pip install --upgrade pip
RUN pip install addict==2.4.0
RUN pip install imageio==2.9.0
RUN pip install kiwisolver==1.3.1
RUN pip install matplotlib==3.3.4
RUN pip install numpy
RUN pip install olefile==0.46
RUN pip install opencv-python==4.5.1.48
RUN pip install Pillow
RUN pip install pyparsing==2.4.7
RUN pip install python-dateutil==2.8.1
RUN pip install PyYAML==5.4.1
RUN pip install six
RUN pip install tqdm==4.58.0
RUN pip install typing-extensions
RUN pip install yacs==0.1.8
RUN pip install yapf==0.30.0
RUN pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install scikit_learn
RUN pip install ordered_set==3.1
RUN pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
RUN pip install thop
RUN pip install transformers
RUN pip install https://download.openmmlab.com/mmcv/dist/1.3.0/torch1.7.0/cu101/mmcv_full-1.3.0%2Btorch1.7.0%2Bcu101-cp38-cp38-manylinux1_x86_64.whl
RUN pip install terminaltables
RUN pip install timm
RUN pip install codecov
RUN pip install flake8
RUN pip install interrogate
RUN pip install isort==4.3.21
RUN pip install pytest
RUN pip install xdoctest>=0.10.0
RUN pip install cityscapesscripts

RUN imageio_download_bin freeimage



 # clean-up
RUN rm -rf /var/lib/apt/lists/*
RUN apt clean && apt autoremove -y

 # provide defaults for the executing container
CMD [ "/bin/bash" ]