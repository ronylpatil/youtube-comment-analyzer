ARG PYTHON_VERSION=3.10.3
FROM python:${PYTHON_VERSION}-slim-buster as base

# redis server details
ARG REDIS_HOST
ARG REDIS_KEY
ENV REDIS_HOST=${REDIS_HOST} REDIS_KEY=${REDIS_KEY}

# Expose the port that the application listens on.
EXPOSE 8000

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE = 1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED = 1

# lightgbm model need parallel processing hence installing
RUN apt-get update && apt-get install -y libgomp1

WORKDIR /app
# instead of copying everying, manually copy only required files in ./app directory to keep the container size as small as possible
# COPY ./prod/docker_client.py ./app/prod/docker_client.py
# COPY ./prod/mlflowdb.py ./app/prod/mlflowdb.py
# COPY docker_requirements.txt ./app/docker_requirements.txt 
COPY . /app    

# Install pip req
COPY docker_requirements.txt .
RUN pip install --no-cache-dir -r docker_requirements.txt 

# Run the application
# 127.0.0.1, it means the service will only be accessible from within the EC2 instance itself, because 127.0.0.1
# refers exclusively to the local machine. To make your service accessible externally, use 0.0.0.0, it listen all ip's
CMD uvicorn api.rest_api:app --host 0.0.0.0 --port 8000
