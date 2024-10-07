# Use a build argument to specify the base image
ARG BASE_IMAGE=rayproject/ray:2.36.0
FROM ${BASE_IMAGE}

USER root

RUN apt-get update -y --fix-missing && apt-get install -y cmake \
                                                          swig \
                                                          zlib1g-dev \
                                                          python3-dev \
                                                          python3-pip \
                                                          clang


RUN pip install torch==2.4.1 \
                pettingzoo[all]==1.24.3 \
                supersuit==3.9.3 \
                gputil==1.4.0

WORKDIR /app