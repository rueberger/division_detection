FROM conda/miniconda2

# System packages
RUN apt-get update && apt-get install -y curl git gcc

RUN conda install -y ipython tensorflow
RUN conda install -y -c ilastik pyklb

# create expected dirs
WORKDIR /results
WORKDIR /logs

# install augment
RUN pip install git+https://github.com/funkey/augment.git

WORKDIR /research

ADD . /research/division_detection

RUN pip install -e division_detection/
