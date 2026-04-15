ARG BASE_IMAGE=ds-segformer:7.1-ready
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream-7.1/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV PKG_CONFIG_PATH=/usr/lib/aarch64-linux-gnu/pkgconfig:/usr/share/pkgconfig

RUN find /etc/apt/sources.list.d -type f -name '*.list' -exec \
    sed -i '/l4t-repo.nvidia.com/s/^/# /' {} + || true

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    libglib2.0-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/fusion_v4l2_deepstream_app

COPY fusion_v4l2_deepstream_app /opt/fusion_v4l2_deepstream_app

RUN rm -rf build

RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --parallel "$(nproc)"

CMD ["/opt/fusion_v4l2_deepstream_app/build/fusion_v4l2_deepstream_app", "--verbose"]
