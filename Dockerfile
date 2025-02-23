# Use the official Python base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

RUN apt-get update && apt-get install -y build-essential wget && \
    tar -xvf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure && \
    make && \ 
    make install && \
    pip install --no-cache-dir TA-Lib

RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Expose port 10000 for the application
EXPOSE 10000

# Command to run your application (example: FastAPI with Uvicorn)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]