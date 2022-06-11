# syntax=docker/dockerfile:1
FROM python:3.10-slim
WORKDIR /app

RUN apt-get update
RUN apt-get install -y git
RUN pip3 install --upgrade pip

COPY requirements.txt .
RUN pip3 install --no-cache-dir \
            Flask==2.1.2 \
            Flask_RESTful==0.3.9 \
            ipython==8.4.0 \
            marshmallow==3.16.0 \
            Pillow==9.1.1 \
            gunicorn==20.1.0 \
            torch==1.11.0 --extra-index-url https://download.pytorch.org/whl/cu113\
            git+https://github.com/openai/glide-text2im

EXPOSE 5000
EXPOSE 8080

COPY . .
RUN python init.py

CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 0 app:app
