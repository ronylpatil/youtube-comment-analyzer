ARG PYTHON_VERSION=3.9.2
FROM python:${PYTHON_VERSION}-slim as base

# Expose the port that the application listens on.
EXPOSE 8000

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE = 1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED = 1

WORKDIR /app
# instead of copying everying, manually copy only required files in ./app directory to keep the container size as small as possible
# COPY ./prod/docker_client.py ./app/prod/docker_client.py
# COPY ./prod/mlflowdb.py ./app/prod/mlflowdb.py
# COPY docker_requirements.txt ./app/docker_requirements.txt 
COPY . /app    

# Install pip req
COPY docker_requirements.txt .
RUN pip install -r docker_requirements.txt 

# Run the application
CMD uvicorn api.rest_api:app --reload --reload-dir ./api --host 127.0.0.1 --port 8000
