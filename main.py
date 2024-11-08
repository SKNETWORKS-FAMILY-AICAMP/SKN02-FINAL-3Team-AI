import dotenv, os, requests

from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware # Colab Local Test 환경

from stt import STT
from sllm import SLLM

dotenv.load_dotenv()

app = FastAPI()
sLLM = SLLM()
stt = STT()

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
    print(f'send_minutes URL ---> {url}')

    response = requests.patch(url, json=params)
    print(f"send_minutes RESULT ---> {response.status_code}")

    generate_summary(meeting_id, content, b_task)

def send_summary(meeting_id: int, summary: str):
    params = {
        "api_key":api_key,
        "meeting_id": meeting_id,
        "summary": summary
    }
    url = os.path.join(web_url, 'meeting/summary')
    print(f'send_summary URL ---> {url}')
    
    response = requests.patch(url, json=params)
    print(f"send_summary RESULT ---> {response.status_code}")

@app.post("/generate_minutes/")
async def generate_minutes(from_django: FromDjango, b_task:BackgroundTasks):
    print("===============================================")
    print("process starts! ========>")
    print("===============================================")
    b_task.add_task(make_minutes, from_django.meeting_id, from_django.audio_url, from_django.num_of_person, b_task)

    response = {
        "message": "Your transcription of meeting will be generated. Please wait..."
    }
    
    return response

@app.post("/generate_summary/")
def generate_summary(meeting_id, content, b_task:BackgroundTasks):
    print("summary starts! ========> meeting_id:", meeting_id)
    b_task.add_task(make_summary, meeting_id, content)

    response = {
        "message": "Your transcription of meeting will be summarized. Please wait..."
    }
    
    return response

def make_minutes(meeting_id: int, audio_url: str, num_of_person: int, b_task:BackgroundTasks):
    print("===============================================")
    print("make_minutes")
    print("===============================================")

    # STT 실행 및 결과 받기
    content=stt.run(audio_url, num_of_person)
    
    send_minutes(meeting_id, content, b_task)

def make_summary(meeting_id: int, content: dict):
    print("make_summary")
    summary = sLLM.sllm_response(content)

    send_summary(meeting_id, summary)