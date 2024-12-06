# If not working, first do: sudo rm -rf /tmp/.docker.xauth
# If still not working, try running the script as root.

XAUTH=/tmp/.docker.xauth

echo "Preparing Xauthority data..."
xauth_list=$(xauth nlist :0 | tail -n 1 | sed -e 's/^..../ffff/')
if [ ! -f $XAUTH ]; then
    if [ ! -z "$xauth_list" ]; then
        echo $xauth_list | xauth -f $XAUTH nmerge -
    else
        touch $XAUTH
    fi
    chmod a+r $XAUTH
fi

echo "Done."
# echo ""
# echo "Verifying file contents:"
# file $XAUTH
# echo "--> It should say \"X11 Xauthority data\"."
# echo ""
# echo "Permissions:"
# ls -FAlh $XAUTH
# echo ""
echo "Running docker..."

# # Check if docker-compose is installed
# if ! command -v docker-compose &> /dev/null; then
#   echo "docker-compose not found, installing..."
#   sudo apt-get update && sudo apt-get install -y docker-compose
# fi

# Build the container only if the image needs to be built
echo "Building the container with Docker Compose..."
docker-compose build

# Run the container
echo "Running the container with Docker Compose..."
docker-compose up -d 
# docker-compose run --service-ports vocal_command /bin/bash -d
# démarre container pour service vocal_command + ouvre un terminal interactif avec /bin/bash
# RUN overrides docker compose so use up instead 

# sudo apt-get install -y docker-compose

# # Get the current working directory
# current_dir=$(pwd)

# # Use dirname to get the parent directory
# parent_dir=$(dirname "$current_dir")

# docker build -f Dockerfile . -t ghcr.io/epflxplore/vocal_command:latest

# # Add this line if you want to use sound: --device /dev/snd your_image_name \ 
# docker run -it \
#     --name base_humble_desktop \
#     --rm \
#     --privileged \
#     --net=host \
#     -e DISPLAY=unix$DISPLAY \
#     -e QT_X11_NO_MITSHM=1 \
#     -e XAUTHORITY=$XAUTH \
#     -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
#     -v $XAUTH:$XAUTH \
#     -v /run/user/1000/at-spi:/run/user/1000/at-spi \
#     -v /dev:/dev \
#     -v $parent_dir:/home/xplore/dev_ws/src \
#     -v base_humble_desktop_home_volume:/home/xplore \
#     --device /dev/snd \
#     ghcr.io/epflxplore/vocal_command:latest

# # pour run docker compose:
# # docker-compose up --build
