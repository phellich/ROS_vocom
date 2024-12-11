################################################
## XPLORE SECURITY AUTH                       ##
################################################

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
# echo "Running docker..."

################################################
## DOCKER CONTAINER BUILD AND RUN             ##
################################################

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
  echo "docker-compose not found, installing..."
  sudo apt-get update && sudo apt-get install -y docker-compose
fi

# Build the container only if the image needs to be built
echo "Building the container with Docker Compose..."
docker compose build

# Up the container (vs Run)
echo "Running the container with Docker Compose..."
docker compose up -d            
# Run overrides docker-compose so use Up instead:
# docker-compose run --service-ports vocal_command /bin/bash -d # start container for vocal_command service and open a terminal with /bin/bash