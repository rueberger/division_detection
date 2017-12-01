FROM tensorflow/tensorflow:latest

# System packages
RUN apt-get update && apt-get install -y curl git

# create expected dirs
WORKDIR /results
WORKDIR /logs

# install augment
RUN pip install git+https://github.com/funkey/augment.git

WORKDIR /src

# install pyklb
# cython is a prereq to compile
RUN pip install cython
RUN git clone https://github.com/bhoeckendorf/pyklb.git && \
    cd pyklb && \
    python setup.py bdist_wheel && \
    pip install dist/*.whl

WORKDIR /research

ADD . /research/division_detection

RUN pip install -e division_detection/
