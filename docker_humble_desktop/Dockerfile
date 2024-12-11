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
RUN pip install --no-cache-dir -r ./requirements.txt && \ 
    rm ./requirements.txt

# Set a directory to build the project
WORKDIR /home/$USERNAME/dev_ws