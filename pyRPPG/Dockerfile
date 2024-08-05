FROM nvidia/cuda:12.4.1-base

WORKDIR /pyRPPG


ADD . /pyRPPG

RUN apt-get clean \
    && apt-get -y update \
    && apt-get -y upgrade \
    && apt-get install -y python3-dev \
    && apt-get install -y python3-pip \
    && apt-get install -y cmake \
    && apt-get install -y nano \
    && apt-get install -y python3-tk \
    && apt-get install -y libsm6 libxext6 libxrender-dev

# pyEDFlib need numpy installed first
RUN pip3 install --upgrade pip \
    && pip3 install numpy==1.16.4 \
    && pip3 install --no-cache-dir -r requirements.txt

