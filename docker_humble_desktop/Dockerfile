###############################################################
# Xplore Human-ROver communication image Build Docker
###############################################################
FROM ghcr.io/epflxplore/base:humble-desktop

# re-utile car on utilise sudo pour installer d'autres packages
ARG DEBIAN_FRONTEND=noninteractive

# Switch to root user to have sufficient privileges for apt-get
USER root
# sudo apt-get install -y alsa-base alsa-utils 
RUN apt-get update && apt-get install -y portaudio19-dev 
# RUN apt-get update && apt-get install -y docker-compose

#utile pour avoir les requirements.txt car supp a la fin de l'autre dockerfile
USER $USERNAME
WORKDIR /home/$USERNAME/dev_ws/src
COPY  requirements.txt requirements.txt

# run line to pip install requirements
RUN pip install -r requirements.txt

# Set a directory to build the project
WORKDIR /home/$USERNAME/dev_ws

EXPOSE 8080

# Clean up
# RUN sudo rm -rf /var/lib/apt/lists/*

# Remove all the confidential Xplore source code from the image (only copied requirements)
# RUN sudo rm -rf /home/$USERNAME/dev_ws/src/*


# To install pyaudio, always need to run 
# sudo apt-get update 
# sudo apt-get install portaudio19-dev
# -y pour rep yes aux prompts

###############################################################
# STUFF DANS IMAGE FROM BASE DONC PAS BESOIN DE REINSTALLER
###############################################################

# # non interactive debian env donc pas de prompt durant le build
# ARG DEBIAN_FRONTEND=noninteractive => deja dans la 1e image donc pas besoin

# # Add USB rules => utile pour microphone???
# RUN echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
# # restart udev service pour appliquer ces USB rules
# RUN /etc/init.d/udev restart