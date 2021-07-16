FROM tensorflow/tensorflow:2.4.1-gpu

ARG USER_ID=1001
ARG GROUP_ID=1001

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user

RUN apt-get update && apt-get install -y --no-install-recommends libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*


COPY repo repo

RUN pip install -r repo/requirements.txt

WORKDIR /workspace
RUN chown user -R /workspace

ENV TFDS_DATA_DIR=/workspace/tensorflow_datasets

USER user