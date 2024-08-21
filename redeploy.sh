#!/bin/bash

# Load variables from .env file
export $(grep -v '^#' .env | xargs)

# Stop and remove the old container
docker stop $CONTAINER_NAME
docker rm $CONTAINER_NAME

# Build the new image
docker build -t $IMAGE_NAME .

# Run the new container
docker run -d --env-file .env --name $CONTAINER_NAME $IMAGE_NAME
