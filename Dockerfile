FROM python:3.10

COPY . /src
WORKDIR /src

RUN sudo apt-get update
RUN sudo apt-get upgrade -y
 
# container에 git 설치
RUN sudo apt-get install git -y
RUN sudo apt-get install ffmpeg

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]