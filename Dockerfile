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
            torch==1.11.0 \
            git+https://github.com/openai/glide-text2im
#RUN pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu113
#RUN pip3 install git+https://github.com/openai/glide-text2im
EXPOSE 5000

COPY . .
RUN python init.py
CMD [ "python", "./app.py" ]