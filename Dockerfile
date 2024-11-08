FROM python:3.10

RUN apt-get update && apt-get install -y sudo
RUN sudo apt-get upgrade -y

# container에 git 설치
RUN sudo apt-get install git -y
RUN sudo apt-get install ffmpeg -y

COPY . /src
WORKDIR /src

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]