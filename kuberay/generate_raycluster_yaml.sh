#!/bin/bash

# Prompt the user for their Docker username
read -p "Enter your Docker username: " DOCKER_USERNAME

DOCKERFILE_NAME="rlx-image.Dockerfile"
IMAGE_NAME="${DOCKERFILE_NAME%%.Dockerfile}"

# Check if the username and image name were provided
if [ -z "$DOCKER_USERNAME" ]; then
    echo "Docker username and image name cannot be empty. Please try again."
    exit 1
fi

# Debug output to verify the username and image name
echo "Using Docker username: $DOCKER_USERNAME"
echo "Using image name: $IMAGE_NAME"

# Wait for one second
sleep 1

# Detect the system architecture
ARCH=$(uname -m)

# Set the BASE_IMAGE based on the architecture
if [ "$ARCH" == "x86_64" ]; then
    BASE_IMAGE="rayproject/ray:2.36.0"
elif [ "$ARCH" == "arm64" ] || [ "$ARCH" == "aarch64" ]; then
    BASE_IMAGE="rayproject/ray:2.36.0-aarch64"
else
    echo "Unsupported architecture: $ARCH"
    exit 1
fi

# Build the Docker image
docker build --build-arg BASE_IMAGE=$BASE_IMAGE -f $DOCKERFILE_NAME -t "$DOCKER_USERNAME/$IMAGE_NAME:latest" .

# Push the Docker image to the user's registry
docker push "$DOCKER_USERNAME/$IMAGE_NAME:latest"

# Remove any existing raycluster.yaml files
rm -f kuberay/raycluster.yaml

# Replace the placeholders in the template file and create the actual YAML file
sed "s/{{DOCKER_USERNAME}}/$DOCKER_USERNAME/g; s/{{IMAGE_NAME}}/$IMAGE_NAME/g" raycluster.yaml.template > raycluster.yaml

# Check if the file was created successfully
if [ -s raycluster.yaml ]; then
    echo "raycluster.yaml generated successfully."
else
    echo "Failed to generate raycluster.yaml. Please check the template and script."
fi