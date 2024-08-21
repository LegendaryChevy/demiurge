#!/bin/bash

# Load variables from .env file
export $(grep -v '^#' .env | xargs)

# Stop the running container
docker stop $CONTAINER_NAME
