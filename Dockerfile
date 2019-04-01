FROM python:3.6-slim

WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 4000

CMD ["gunicorn", "-b", "0.0.0.0:4000", "app"]
