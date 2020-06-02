from tensorflow/tensorflow:1.12.0-devel-gpu-py3

# Install python3.6
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update
RUN apt install -y python3.6
RUN rm /usr/bin/python
RUN rm /usr/bin/python3
RUN ln -s /usr/bin/python3.6 /usr/bin/python
RUN ln -s /usr/bin/python3.6 /usr/bin/python3

RUN apt remove python3-pip
RUN apt install -y python3-pip
RUN apt install -y libpython3.6-dev

CMD python --version

RUN mkdir -p /mnt/workspace/src
WORKDIR /mnt/workspace/src
COPY . /mnt/workspace/src

RUN python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# RUN python -m pip uninstall --yes tensorflow
# RUN python -m pip install tensorflow-gpu==1.12.0

# Support Chinese
ENV LANG C.UTF-8

ENTRYPOINT  ["python", "predictor.py"]