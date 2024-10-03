# if you are using an AMD64-processor (most windows/linux computers)
FROM rayproject/ray:2.36.0

# if you are using an ARM64-processor (apple computers with M-chips)
#FROM rayproject/ray:2.36.0-aarch64

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