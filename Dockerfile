FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    gcc \
    g++ \
    make \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir pytest pandas numpy

WORKDIR /neuro-cdcl
COPY . /neuro-cdcl

RUN make build-all

CMD ["make", "reproduce"]
