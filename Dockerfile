# Use the official Python image from the DockerHub
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make sure the scripts are executable
RUN chmod +x /app/run_scripts.sh

# Command to run the scripts in order
CMD ["./run_scripts.sh"]
