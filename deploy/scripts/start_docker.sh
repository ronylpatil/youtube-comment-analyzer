#!/bin/bash
# Log everything to start_docker.log
exec > /home/ubuntu/start_docker.log 2>&1

echo "Logging in to ECR..."
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 471112948185.dkr.ecr.us-east-1.amazonaws.com

echo "Pulling Docker image..."
docker pull 471112948185.dkr.ecr.us-east-1.amazonaws.com/youtube-chrome-plugin:latest

echo "Checking for existing container..."
if [ "$(docker ps -q -f name=ytca-app)" ]; then
    echo "Stopping existing container..."
    docker stop ytca-app
fi

if [ "$(docker ps -aq -f name=ytca-app)" ]; then
    echo "Removing existing container..."
    docker rm ytca-app
fi

# container name: ytca-app
echo "Starting new container..."
if [ -f .env ]; then
    # run docker container and pass .env
    docker run -d -p 80:8000 --env-file .env --name ytca-app 471112948185.dkr.ecr.us-east-1.amazonaws.com/youtube-chrome-plugin:latest    
else
    echo ".env file not found. Please ensure its created..."
    exit 1
fi
# docker run -d -p 80:8000 --name ytca-app 471112948185.dkr.ecr.us-east-1.amazonaws.com/youtube-chrome-plugin:latest

echo "Container started successfully."
