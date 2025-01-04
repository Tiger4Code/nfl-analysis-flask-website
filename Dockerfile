FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the code to the container
COPY requirements.txt /app/requirements.txt

# Install the necessary dependencies
RUN pip install -r requirements.txt

# Command to run the nfl application
CMD ["python", "app.py"]
