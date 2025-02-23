# Use the official Python base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Update pip and install system dependencies needed for TA-Lib
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    libta-lib0 \
    libta-lib-dev

# Install Python dependencies from requirements.txt
RUN pip install -r requirements.txt

# Expose port 10000 for the application
EXPOSE 10000

# Command to run your application (example: FastAPI with Uvicorn)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]