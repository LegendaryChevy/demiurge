#!/bin/bash

# Load variables from .env file
export $(grep -v '^#' .env | xargs)

# Stop the running container
docker start $CONTAINER_NAME
