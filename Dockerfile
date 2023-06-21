FROM nvidia/cuda:11.3.0-base-ubuntu20.04

# RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

# RUN wget -qO- https://get.docker.com/gpg | sudo apt-key add -
USER root

ENV TZ=Europe
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y wget  \
    && apt-get install -y git-all \
    && apt-get install python3.8 -y \
    && apt-get install python3.8-distutils  -y  \
    && ln -s /usr/bin/python3.8 /usr/bin/python
# RUN apt-get update&& \
#     apt-get install -y software-properties-common && \
#     add-apt-repository -y ppa:deadsnakes/ppa && \
#     apt-get update && \ apt-get install python3.8 -y  

RUN python --version


RUN wget https://bootstrap.pypa.io/get-pip.py

RUN python get-pip.py

COPY requirements.txt /root/requirements.txt

RUN python -m pip install -r /root/requirements.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org
