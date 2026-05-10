# GPU FEniCS (CUDA 12.5 + PETSc 3.24 + DOLFIN 2019.1.0 + CuPy 13.3) Dockerfile
# Reproduces the environment described in GPU_FEniCS_Installation_Manual.md
#
# Build:
#   docker build -t gpu-fenics:latest .
#
# Run:
#   docker run -it --gpus all --shm-size=16g \
#     -v /path/to/work:/home/work \
#     --name gpu-fenics gpu-fenics:latest

FROM nvidia/cuda:12.5.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PETSC_DIR=/opt/petsc
ENV PETSC_ARCH=arch-linux-c-opt
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ENV PYTHONPATH=/usr/local/lib/python3.10/dist-packages
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV PETSC_OPTIONS="-mat_type aijcusparse -vec_type cuda -use_gpu_aware_mpi 0"
ENV CUDA_VISIBLE_DEVICES=0,1

# ---- Step 2: System packages ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        gfortran \
        git \
        wget \
        curl \
        pkg-config \
        python3 \
        python3-dev \
        python3-pip \
        python3-setuptools \
        libopenmpi-dev \
        openmpi-bin \
        openmpi-common \
        libhdf5-openmpi-dev \
        hdf5-tools \
        libboost-dev \
        libboost-filesystem-dev \
        libboost-iostreams-dev \
        libboost-program-options-dev \
        libboost-system-dev \
        libboost-thread-dev \
        libboost-timer-dev \
        libeigen3-dev \
        libptscotch-dev \
        libscotch-dev \
        libsuitesparse-dev \
        libxml2-dev \
        libcholmod3 \
        libumfpack5 \
        zlib1g-dev \
        pybind11-dev \
    && rm -rf /var/lib/apt/lists/*

# ---- Step 3: Python base packages ----
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir setuptools==59.6.0 cython && \
    pip3 install --no-cache-dir \
        numpy==1.26.4 \
        scipy==1.15.3 \
        sympy==1.11.1 \
        mpi4py==4.1.1 \
        pybind11==2.2.4 \
        pkgconfig==1.5.5 \
        lark==1.3.0

# ---- Step 4: PETSc 3.24.0 (CUDA + hypre + metis + parmetis) ----
RUN cd /opt && \
    git clone https://gitlab.com/petsc/petsc.git && \
    cd petsc && \
    git checkout v3.24.0 && \
    python3 ./configure \
        --with-cc=mpicc \
        --with-cxx=mpicxx \
        --with-fc=0 \
        --with-cuda=1 \
        --with-cuda-dir=/usr/local/cuda \
        --with-cudac=/usr/local/cuda/bin/nvcc \
	--with-cuda-arch=89 \
        --download-f2cblaslapack=1 \
        --download-hypre=1 \
        --download-metis=1 \
        --download-parmetis=1 \
        --with-debugging=0 \
        PETSC_ARCH=arch-linux-c-opt && \
    make PETSC_DIR=/opt/petsc PETSC_ARCH=arch-linux-c-opt all -j"$(nproc)"

# ---- Step 5: petsc4py ----
RUN pip3 install --no-cache-dir petsc4py==3.24.0 --no-build-isolation

# ---- Step 6: FEniCS Python components (UFL/FIAT/dijitso/FFC) ----
RUN pip3 install --no-cache-dir \
        fenics-ufl==2019.1.0 \
        fenics-fiat==2019.1.0 \
        fenics-dijitso==2019.1.0 && \
    pip3 install --no-cache-dir \
        git+https://bitbucket.org/fenics-project/ffc.git@a799b74378a06dde24f726377574ab7a7973d871

# ---- Step 7: DOLFIN 2019.1.0 + GPU MPI Ghost Vec patch ----
COPY gpu_ghost_full.patch /tmp/gpu_ghost_full.patch

RUN cd /opt && \
    git clone https://bitbucket.org/fenics-project/dolfin.git && \
    cd dolfin && \
    git checkout 74d7efe1e84d65e9433fd96c50f1d278fa3e3f3f && \
    # 7-2 (A): Boost endian compatibility shim (Boost 1.74+)
    sed -i 's|#include <boost/detail/endian.hpp>|#include <boost/predef/other/endian.h>\n\n\n\n// --- Boost endian compatibility shim for old DOLFIN code ---\n#ifndef BOOST_LITTLE_ENDIAN\n  #if defined(BOOST_ENDIAN_BIG_BYTE) \&\& BOOST_ENDIAN_BIG_BYTE\n    #ifndef BOOST_BIG_ENDIAN\n      #define BOOST_BIG_ENDIAN 1\n    #endif\n  #else\n    #ifndef BOOST_LITTLE_ENDIAN\n      #define BOOST_LITTLE_ENDIAN 1\n    #endif\n  #endif\n#endif|' \
        dolfin/io/VTKFile.cpp dolfin/io/VTKWriter.cpp && \
    # 7-2 (B): GCC 11+ requires <algorithm> for std::count, std::min_element
    sed -i '1i #include <algorithm>' dolfin/mesh/MeshFunction.h && \
    sed -i '1i #include <algorithm>' dolfin/geometry/IntersectionConstruction.cpp && \
    # 7-3: GPU MPI Ghost Vec full patch
    git apply /tmp/gpu_ghost_full.patch && \
    # 7-4: Build DOLFIN C++
    mkdir build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_SHARED_LIBS=ON \
        -DDOLFIN_ENABLE_MPI=ON \
        -DDOLFIN_ENABLE_PETSC=ON \
        -DDOLFIN_ENABLE_HDF5=ON \
        -DDOLFIN_ENABLE_SCOTCH=ON \
        -DDOLFIN_ENABLE_PARMETIS=ON \
        -DDOLFIN_ENABLE_CHOLMOD=ON \
        -DDOLFIN_ENABLE_UMFPACK=ON \
        -DDOLFIN_ENABLE_ZLIB=ON \
        -DDOLFIN_ENABLE_PYTHON=ON && \
    make -j"$(nproc)" && \
    make install && \
    ldconfig && \
    # 7-5: DOLFIN Python bindings
    cd /opt/dolfin/python && \
    pip3 install --no-cache-dir --no-build-isolation . && \
    rm /tmp/gpu_ghost_full.patch

# ---- Step 8: CuPy and extras ----
RUN pip3 install --no-cache-dir \
        cupy-cuda12x==13.3.0 \
        pyamg==5.3.0 \
        matplotlib==3.10.8 \
        jupyterlab==4.4.9 \
        notebook==7.4.7 \
        nvidia-ml-py==13.580.82

# ---- Step 9: Shell environment (interactive bash sessions) ----
RUN printf '%s\n' \
    '' \
    '# === GPU FEniCS Environment ===' \
    'export OMPI_ALLOW_RUN_AS_ROOT=1' \
    'export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1' \
    'export PETSC_DIR=/opt/petsc' \
    'export PETSC_ARCH=arch-linux-c-opt' \
    'export PYTHONPATH=/usr/local/lib/python3.10/dist-packages:$PYTHONPATH' \
    'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' \
    'export PETSC_OPTIONS="-mat_type aijcusparse -vec_type cuda -use_gpu_aware_mpi 0"' \
    'export CUDA_VISIBLE_DEVICES=0,1' \
    '' \
    '# NVIDIA MPS auto-start (multi-GPU efficiency)' \
    'if command -v nvidia-cuda-mps-control &>/dev/null; then' \
    '    if ! pgrep -x nvidia-cuda-mps-control &>/dev/null; then' \
    '        nvidia-cuda-mps-control -d' \
    '        echo "[MPS] daemon started (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all})"' \
    '    else' \
    '        echo "[MPS] daemon already running"' \
    '    fi' \
    'fi' \
    >> /root/.bashrc

WORKDIR /home/work
CMD ["/bin/bash"]

