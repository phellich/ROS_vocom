###########################################
# Xplore Human-ROver communication image Build Docker
###########################################
FROM ghcr.io/epflxplore/base:humble-desktop

# To install pyaudio, always need to run 
# sudo apt-get update 
# sudo apt-get install portaudio19-dev
RUN apt-get update && \
    apt-get install -y portaudio19-dev

# run line to pip install requirements
RUN pip install -r requirements.txt
