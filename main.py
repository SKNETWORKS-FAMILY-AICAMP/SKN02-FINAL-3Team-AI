import dotenv, os, requests, uvicorn, logging

from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware # Colab Local Test 환경

from stt import STT
from sllm import SLLM

dotenv.load_dotenv()

app = FastAPI()
sLLM = SLLM()
stt = STT()

logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.DEBUG)

# Colab Local Test 환경
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

api_key = os.environ['api_key']
web_url = os.environ['web_url']

class FromDjango(BaseModel):
    meeting_id: int
    audio_url: str
    num_of_person: int


@app.get("/")
def hello():
    return {"message": "Hello world!"}

def send_minutes(meeting_id: int, content: dict, b_task:BackgroundTasks):
    params = {
        "api_key":api_key,
        "meeting_id": meeting_id,
        "content": content
    }
    
    url = os.path.join(web_url, 'meeting/detail')
    logger.debug(f'send_minutes URL ---> {url}')

    response = requests.patch(url, json=params)
    logger.debug(f"send_minutes RESULT ---> {response.status_code}")

    generate_summary(meeting_id, content, b_task)

def send_summary(meeting_id: int, summary: str):
    params = {
        "api_key":api_key,
        "meeting_id": meeting_id,
        "summary": summary
    }
    url = os.path.join(web_url, 'meeting/summary')
    logger.debug(f'send_summary URL ---> {url}')
    
    response = requests.patch(url, json=params)
    logger.debug(f"send_summary RESULT ---> {response.status_code}")

@app.post("/generate_minutes/")
def generate_minutes(from_django: FromDjango, b_task:BackgroundTasks):
    logger.debug("===============================================")
    logger.debug("process starts! ========>")
    logger.debug("===============================================")
    b_task.add_task(make_minutes, from_django.meeting_id, from_django.audio_url, from_django.num_of_person, b_task)

    response = {
        "message": "Your transcription of meeting will be generated. Please wait..."
    }
    
    return response

@app.post("/generate_summary/")
def generate_summary(meeting_id, content, b_task:BackgroundTasks):
    logger.debug(f"summary starts! ========> meeting_id: {meeting_id}")
    b_task.add_task(make_summary, meeting_id, content)

    response = {
        "message": "Your transcription of meeting will be summarized. Please wait..."
    }
    
    return response

def make_minutes(meeting_id: int, audio_url: str, num_of_person: int, b_task:BackgroundTasks):
    logger.debug("===============================================")
    logger.debug("make_minutes")
    logger.debug("===============================================")

    # STT 실행 및 결과 받기
    content=stt.run(audio_url, num_of_person)
    
    send_minutes(meeting_id, content, b_task)

def make_summary(meeting_id: int, content: dict):
    logger.debug("make_summary")
    summary = sLLM.sllm_response(content)

    send_summary(meeting_id, summary)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)