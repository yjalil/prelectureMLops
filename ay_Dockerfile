FROM python:3.10.6-slim AS python


RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  libpq-dev \
  && rm -rf /var/lib/apt/lists/*


# Set work directory
WORKDIR /app

# Install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy project
COPY . /app/

RUN chmod 755 /app

# Run entrypoint
ENTRYPOINT ["python", "-m", "project.main", "-run", "model3"]
