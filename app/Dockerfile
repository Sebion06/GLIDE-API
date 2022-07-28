# syntax=docker/dockerfile:1
FROM python:3.10-slim
WORKDIR /app

RUN apt-get update
RUN apt-get install -y git
RUN pip3 install --upgrade pip

COPY requirements.txt .
RUN pip install -r requirements.txt

EXPOSE 5000
EXPOSE 8080

COPY . .
RUN python init.py

CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 0 app:app
