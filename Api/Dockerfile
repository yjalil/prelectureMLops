FROM python:3.11.9-slim AS python

# Set work directory
WORKDIR /app

# Install dependencies
COPY . .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8000

# Run entrypoint
ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0","--port","8000"]
# build -t xxx build image
