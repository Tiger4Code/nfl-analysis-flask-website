FROM python:3.13-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the code to the container
COPY requirements.txt /app/requirements.txt

# Install the necessary dependencies
RUN pip install -r requirements.txt

# Copy the code to the container
COPY . /app/

# Command to run the nfl application
CMD ["python", "app.py"]
