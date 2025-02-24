###############################################################
# Xplore Human-ROver communication image Build Docker
###############################################################
# Checking size of layers: docker history  docker_humble_desktop_vocal_command
# Detail informations:  docker image inspect docker_humble_desktop_vocal_command

# Xplore Recruitment image 
FROM ghcr.io/epflxplore/base:humble-desktop 

# Switch to root user to have sufficient privileges for apt-get
USER root

# Prepare LlamaCpp and PyAudio installation, Intsall PulseAudio
RUN apt-get update && \
    apt-get install -y \
        portaudio19-dev \
        pulseaudio pulseaudio-utils \
        ninja-build \
        cmake \
        build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
# sudo apt-get install -y alsa-base alsa-utils 

# Install requirements (uv in docker: https://docs.astral.sh/uv/guides/integration/docker/#installing-uv)
USER $USERNAME
WORKDIR /home/$USERNAME/dev_ws/src
COPY dev_ws/src /home/$USERNAME/dev_ws/src
# COPY dev_ws/src/requirements.txt ./requirements.txt
# RUN pip install --no-cache-dir -r ./requirements.txt && \ 
#     rm ./requirements.txt
RUN pip install -r ./requirements.txt && \ 
    rm ./requirements.txt

# Install LlamaCpp
RUN pip download llama-cpp-python && \
    tar -xvf llama_cpp_python-*.tar.gz && \
    cd llama_cpp_python-* && \
    sed -E -i 's/-mcpu=native\+nodotprod\+noi8mm\+nosve/-mcpu=cortex-a57/g' vendor/llama.cpp/ggml/src/ggml-cpu/CMakeLists.txt && \
    # Vérifier que le remplacement a bien été effectué (optionnel)
    grep -R "cortex-a57" vendor/llama.cpp/ggml/src/ggml-cpu/CMakeLists.txt && \
    pip install . && \
    cd .. && rm -rf llama_cpp_python-*



RUN pip install . && pip install --force-reinstall numpy==1.24.4

# Set a directory to build the project
WORKDIR /home/$USERNAME/dev_ws
