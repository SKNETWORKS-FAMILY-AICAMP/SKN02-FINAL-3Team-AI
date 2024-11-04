import uvicorn
from fastapi import FastAPI, BackgroundTasks
import dotenv, os, requests, time
from pydantic import BaseModel

from sllm import sLLM

app = FastAPI()
sLLM = sLLM()

dotenv.load_dotenv()

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
    
    response = requests.patch(os.path.join(web_url, 'meeting/detail'), json=params)
    print("send_minutes ==> ", response.status_code)

    generate_summary(meeting_id, content, b_task)

def send_summary(meeting_id: int, summary: str):
    params = {
        "api_key":api_key,
        "meeting_id": meeting_id,
        "summary": summary
    }

    response = requests.patch(os.path.join(web_url, 'meeting/summary'), json=params)
    print("send_minutes ==> ", response.status_code)

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
    time.sleep(2)
    content = {
        "minutes": [
            {
                "speaker": "SPEAKER_04",
                "text": " 그냥 저기에 내가 가려둔"
            },
            {
                "speaker": "SPEAKER_03",
                "text": " 그냥 가려둔"
            },
            {
                "speaker": "SPEAKER_00",
                "text": " 8시 4분?"
            },
            {
                "speaker": "SPEAKER_03",
                "text": " 에스카이 사이토닝 결과"
            },
            {
                "speaker": "알 수 없음",
                "text": " 어쩌저쩌고 해갖고 확인하고"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그래서 블라썩을 적용한 결과까지 나오는 거 보죠?"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 네 맞습니다"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 이렇게 해가지고 뒤로 해서"
            },
            {
                "speaker": "알 수 없음",
                "text": " 대충 좀 정치려고 뭐야"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그래서 얘를 쓰니까 좀 더 낫는데"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 또 불구하고 또 내리면"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 브레인스톤이 앞에 계시던데"
            },
            {
                "speaker": "SPEAKER_00",
                "text": " 이게 지금 이상한 애가 반복돼서 나오는 게 뭔가 있는 거 아니야?"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 네 이거는 이제 긴 거를 한 번에 넣었을 때"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 이게 앞에는 몇 분짜리?"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 앞에는 약 30분 정도고요"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 한 시간짜리도 반만 온가?"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 네 뒤에 게"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 한 시간 반짜리?"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 네 한 시간은 아직"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 네 데이퍼가 없어서"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 우리 다음번에 아예"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 지금 이 모델을 할지는 모르지만"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 주말에 모이면"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 주말에 모이거나"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 이거 할 때"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 한 번 녹음 한 번"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 한 시간짜리 되잖아"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 해 놓고"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그것도 없는 거"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 아니면 여러분들 강의를 할 때"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 대부분"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 아 근데 강의"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 강의 시간에 우리끼리"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 회의를 하면 되잖아"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 회의를 하잖아"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 그러면 끝나고 쉬는 시간 있고"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 이렇게 해주는 거 아니야?"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 네 그래서 지금"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 제가 가장 베스트인 거는"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 레그를 안 쓰면 저는 좋을 거 같기는 해요"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 이게 왜?"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 왜냐면은"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 레그를 넘어가면은"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 레그에서 일단은 중요하다고"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 판단하는 부분을 이제 또 뽑아서"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 그 부분에 대한 요약을 진행을 하는데"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 그러니까 그게 스플릿된 데이터를 가지고"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 거기에서 요약을 진행을 하는데"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 레그에서 만약에 잠깐 지나간 말을"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 이걸 중요하지 않다고"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 인식을 해 버릴 가능성이 조금 있어서"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 어차피 웨이팅을 어디에 찍느냐에 따라"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 정확하게 따진다는"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 그런 것보다는"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 일단은 전체 맥락을"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 일단 다 담아서 표현해 줄 수 있는 게"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 소주로 진행하니까"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 네 그래서 이제 한 시간을 넣어봤을 때"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 만약에 잘 나온다라고 하면은"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 저런 위에 보면은"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그런 마케팅을 이상하게 신다던가"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 저런 문제를 해결하기 위해서는"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 한국어에 대한 요약 데이터나 그런 거를"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 조금 더 파이트닝을 조금 더 시켜주면"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 잡을 수 있을 것 같아서요"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 네 그렇습니다"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 그래서 일단은 레그 방법이 있기는 하지만"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 지금 방법으로는 파인튜닝 된 모델 그대로를 써서"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 조금 더 파인튜닝을 익히는 방법으로 가고 싶어요"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 일단 지금 건 나쁘지 않은 것 같아요"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 그래서 한 방으로"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 추천레이션이 너무 커질 것 같아요"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 활용할 수 있냐라고 하는"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 이거 자체로 학습시키고 하는데"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 얼마라고 해요?"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 모델 학습시키는 거는"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 한 10..."
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 에프크다"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그냥 에프크 해서"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 몇 가지 공부하라고 하면"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 12시간"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 일단은 기본적으로"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 사회 이상으로 가요"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 네"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 저 모델 하나에"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 에프크를 열 번을 돌렸는데"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 하나만 더 해보면"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 지금 바로 말한 대로"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 레그 쓰지 말고"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 방금 말한 방식에"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 파인튜닝 들어가는 거"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 근데 그건 지금 우리가"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 논"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 내가 올려줬던 거"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 이게 약간"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 다국어 이슈 있는 부분에서"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 좀 더 나은 걸 써보면 어떨까"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 라는 생각이 들기는 해서"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 제 한 번 모델을 써볼까"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 저는 좋은"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 한번 거기에다가"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그게 그냥 나와서"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 괜찮으면 그냥 해도 되는데"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 조금 이상하면"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그냥 골랐으면 쓰고"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그럼 한 번 정도면"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 해볼 수 있잖아요"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 네"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 저 이거 내일 돌려놓고"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 일단 확인을 해 볼 수 있어요"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그렇게 해보시죠"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 두 가지로 해서"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그러면 그중에"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그중에 이제"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 의사결정하면 돼요"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 그중에 나온 걸로 쓰면 돼요"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그리고 이거를"
            },
            {
                "speaker": "알 수 없음",
                "text": " 지금 모델 하나 세 번 정도 한 거잖아요"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 우리가 그것까지 하면"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 세 개 정도 쓰는 거잖아"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그럼 그걸 다"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 이 모델의 변천사나 이런 부분들"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 우리 해 본 거에다가"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 나올 수 있어요"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 그리고 방금 나온 모델은"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 거의 최신 모델인데"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 그 이 모델은"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 지금"
            },
            {
                "speaker": "알 수 없음",
                "text": " 보내주셨던 게"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 80억"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 80억"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 파라미터"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 내가 그거"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 도전에다가"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 좀 더 정리를 해서 올려놔"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 아 네 감사합니다"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 한국어에서 특히 지금"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 거의 한"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 40에서 50% 이상"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 성향성이 좀 있는 거 같아"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그러면"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 아무튼"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 뭐라도"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 이건 내가 저 안에 페이지에다가"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 2b"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 아 이거"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 아야 익스펜스 모델"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 아 맞아요"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 아야 30억"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 익스펜스 32b인데"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 이게 두 개 버전으로 나와 있어서요"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 이게 8b짜리 하나하고"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 32b짜리가 하나 있는 거 같아요"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 지금 아마 비교 붙인 애들은"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 8b짜리인 거 같네요"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 아 네"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 아야 23에 8b다"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 근데 이게 컨텍스트 사이즈가"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 지금"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 8k인 거 같아서요"
            },
            {
                "speaker": "알 수 없음",
                "text": " 있는 게"
            },
            {
                "speaker": "알 수 없음",
                "text": " 네 근데 이게 8k면은"
            },
            {
                "speaker": "알 수 없음",
                "text": " 안 되겠네"
            },
            {
                "speaker": "알 수 없음",
                "text": " 네"
            },
            {
                "speaker": "알 수 없음",
                "text": " 조금"
            },
            {
                "speaker": "알 수 없음",
                "text": " 너무 짧으네 윈도우가"
            },
            {
                "speaker": "알 수 없음",
                "text": " 네"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 32b짜리는"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 8b에서 지금 35b까지 나왔거든"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 아 이게 32b짜리는"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 128k가 나오긴 합니다"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 그치"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 네"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 근데 조금 더 걸린다"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 네"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 사이즈가"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 이게"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 근데 사실 그렇게 우리 거"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 의미가 있잖아"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 네"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 차지 없어요"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 그러면 이거"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 33b짜리도 쓸 수 있는지 한번 확인을 해봐야 되거든요"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 32b짜리를"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그래도"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 하이튜닝을 시도라도 해보고"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그러니까"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 네 우선 이게"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 쓸 수 있으면"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 네"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 한번 해보고"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 유명한 게"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 유명한 게 나오는지 안 나오는지 확인 한번 해보고"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 네"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 일단 API 레벨에서는"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 8선도 지원을 하니까"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 일단은 이게"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 아무래도"
            },
            {
                "speaker": "알 수 없음",
                "text": " 파라미터가 조금은"
            },
            {
                "speaker": "알 수 없음",
                "text": " 커서"
            },
            {
                "speaker": "SPEAKER_03",
                "text": " 조금"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그니까"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그냥 지금"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 성능 테스트 한 거여서"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 영어"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 일본어"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 한국어"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 너는 거여서"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 해 볼 만 하긴 하겠죠"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 메모리 인슈가 없을까 걱정은 되기는 해요"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 하하"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 그것도 다른 인슈"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 그것도 다른 이슈"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 이게"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 파인튜닝 할때 메모리 인슈가 있을까봐"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 조금"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 터지면 어떻게 해?"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그냥 아예 학습이 안 돼"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그냥 힐이야?"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 네"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " OOM 딱 뜨고"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 처음부터"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 이게"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 배치사이즈 조절을 할 수 있는데"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 이게 아무리 줄여도"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 기본적으로"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 펴낼 때랑 펴낼 때"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 펴낼 못 한다는 거네"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 네네"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 옵션이 지금 얘만 있는 건 아니니까"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 얘 걸로 해서 나타날 수 있는지 확인을 해보고"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 그러면은 아야로 32V짜리로 한번 해보겠습니다"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 일단은 1차로 해야 되는 거는"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 지금 한 거 먼저 파인튜닝 하고"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 그 결과를 보고 안전빵 하나 남겨놔야 돼요"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 얘는 이래서 이렇게 나머지 나머지 하나 해놓고"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 거기에서 더 나간 2종의 얘를 하나 하면"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 약간 스토리가 더 잘 만들어질 수 있고"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 이런 것도 해봤는데"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 여기에서 요거는 어려운 이런 게 있다"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 확인하고 그러면 사실상"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 해야 될 것들과 해야 되는 것들"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 해야 될 것들에서"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 목표를 했던 부분들은 어느 정도 좋네"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 생각의 케이스는 일단 면하긴 한 거 같아요"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 신경 써가지고 수고를 해줘서"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 지난번 쓸 때랑 상황이 달라지기는"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그러면 오후 테스트까지"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 일단 금요일까지는 아니어도"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 주말까지는 가능하니까"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그러면 일단 저는 블러썸 파인튜닝 한 것을"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 재파인튜닝을 하고 우선"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 랙 제거하고"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 한 다음에 내용적으로 피로 한번 해주고요"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그 다음에 여기 들어가면 되죠"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 아야롱"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 아야롱"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 아야롱"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 아야롱"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 이런 부분에서 뭔가 인식이 있거나 이런 부분에서"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 지금 부트스트랩으로 해서"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 모달창으로"
            },
            {
                "speaker": "알 수 없음",
                "text": " 전체 회의록을 띄워줄 생각이었는데요"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 거기가 이제 창이 안 뜨는 문제가 지금 발생하고 있어서"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 왜지?"
            },
            {
                "speaker": "SPEAKER_01",
                "text": "찍어봤어요?"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 불러오기를 해?"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 호출을 하는 게 맞아?"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그럼 만약에 저걸 다시 해봤을 때"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 초출의 문제는 아니에요"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 이게 왜냐하면 회의록 요약에서 같이 불러올 수 있는 내용들이고"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 안 돼서 뷰로 처리를 하면은"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 새 창에서 결국에는 세부 화면을 보는 거잖아요"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 거기에 누르면 그냥 모달창을 보여줬다 안 보여줬다만 하면 되는데"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 세부 회의로 보는 거잖아요"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 얘가 디테일로 들어가는 것처럼 만들 수 있는 거"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 모달의 내용은 있는데"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 다르게 얘기하면 뷰가 이동이 되는데"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 이제 JS가 작동을 하지 않고 있어요"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 들고는 움직이는 거"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그럼 이제 다시 돌아가기를 했을 때"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그 얘기는 뭐냐면"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 폴은 지금 이루어졌는데"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 렌더링이 안 된다는데"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 남아 있어야지"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 그게 뭔가 부트스트랩에서 뭔가 초출이 있는 거 같아요"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 그러니까 그걸 우리가 정의하게 나름대로"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 우리가 어떻게 정의하는지"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 백엔드의 문제가 아니라"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 나 한번 봅시다"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 지금까지 뛰어봐"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 그냥 프런트에서"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 모다리 지금 안 뜬다고"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 다른 캐시를 잡아도 상관없어"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 렌더링이 좀"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 되는 거 그냥 어떤 방법인지"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 자바스피트 펑션이 동작을 했는데 자바스피트 못 불러왔다는 거"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 얘가 가면 뜰 떠야 되는 건데"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그러면 기본적으로 자바스피트에서"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 그럼 이제 여기서 이제"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 세부 정의한다고 하면"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 버전 안 맞거나"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 여기에 이렇게 해서 얘가"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 못 불러오는 라이브로젠이 있거나"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 무달로 안 뜨면은"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 그런 건 아니고"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 뷰를 어떻게 고려해야 되는지"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 에러가 뜨는 리스트는 아예 없죠?"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 아니 저 미달의 뷰를 어떻게 고려한 건가요?"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 로그에서?"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 아니면"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 새로운 뷰로 표시되게 해버린 거야"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 좀 신기한 그"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 로그인 페이지에서는 제대로 언론이 떠서"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 그냥 하나가 날라가고"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 이제 뷰를 우리가 정해야 돼"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 저장 과정이 있었어요"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 아니면은 지금 생각에는"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 이거 그거일 수 있는 거"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 그냥 떠 있는 거 아니야?"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 이렇게 확실하게 날라진 게 나"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 밑에 그냥 쭉 보여주는 방법"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 이거를 들어가자마자"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 사실 나쁘지 않지"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 그냥 놀고 있는 공간이라서"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 바로 로그인이 아니고"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 녹음할 때는"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그냥 얘를 밑에 되게 펑을 때"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 문제가 없이 되는 거예요?"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 전혀 나쁘지 않아"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 이걸 적었다가 열었다가"
            },
            {
                "speaker": "알 수 없음",
                "text": " 그때 처음 띄워?"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 그때 띄우든 나중에 띄울 것 같아요"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 그러니까 그때 처음 띄워?"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 많이 길어지던데요?"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 그걸 할 때"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 이 모달이 그때 처음 띄는데요?"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 뭐 그래도"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 근데 어차피 길어지는 건"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그러니까 모달이라는 애들은"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 모다리도 똑같이 길어졌어"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 사실 윈도우가 뜨면"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그 길이는 모다리라고"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 이게 지금 여기에 떠 있는 걸로 인식을 했는데"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 뭔가 바뀌잖아"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 어차피 모다리 여기 그냥"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 얘가"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 이 공간만 차지했던 거지"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 편하지 않죠?"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 얘가"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 길이 내려가는 거는"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 저기가 안 되면"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그냥 떠 있는 걸로 인식이 돼서"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 얘 만큼만 지금 없고"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 좁아진다고 해서"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 몇 개씩 페이지 메이지를 한다는"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 다시 뜨고 있지 않다는 게 된다는 거예요"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그런 만큼 가져가고"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 리스트가 아니죠?"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그거랑"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 로달 안 찌르고"
            },
            {
                "speaker": "알 수 없음",
                "text": " 제가 어차피 회의록이 딱 돼서"
            },
            {
                "speaker": "알 수 없음",
                "text": " 하자마자"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그냥 샥 만들어지네"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 바로 정답을 봐서"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 얘를 띄우려고 해도 안 돼서"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 근데 보기를 하긴 해야 되겠다"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그러니까 결국에는"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 처음"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그러니까 내가 궁금한다면"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 만약에 30분짜리 넣었을 때"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 처음에 떴는데"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그 다음에 안 뜨는 건지"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 스크롤 얼마나"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그런 문제는"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 이렇게 해서 나타나는지"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그게 아니라는 건지"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 구성도"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 1시간 반짜리 해서"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 만약에 이런 거예요"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 얼마나 나타나는지 한번"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그러니까 이런 모달 같은 이슈가"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 보기를 해야 될지"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 굉장히 어떻게 되는지 아닌지"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 근데 뭐"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 우리가 되게 퀴수가 아니면"
            },
            {
                "speaker": "SPEAKER_04",
                "text": " 불거 자체가 되게"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그냥 퀴수로 처리해"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 패턴하지 않나요?"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 아예 새 화면을"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 문제만 하면 불러오기만 하면 돼"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그냥 퀴수로 처리해"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그러니까"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 이거 갖고"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 이런 느낌으로"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 뭔가 시원한 게 없잖아"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그냥 그냥"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그냥"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그치 그치"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 되게 나이스하지 않아도"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 어 나쁘지 않아 나쁘지 않아"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 대충 다른 거는 띄울 수 있다"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 어 나쁘지 않아"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그러면 그걸로 하면"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 이제"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 기회로 처리해 봐야 돼"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 기술적으로"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 근데 뭔가 느낌은"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 흐르고 있다는 느낌이"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 자바 스크립 호출을 할 때"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 이게 정상적으로"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 지금 워킹하고"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 스테이킹이나 그런 정도의"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 근데 그게 뭔가"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 열고 닫고 열고 닫고 할 때"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그 쟤는"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 칠칠칠 핸낮은 왜 뜨는 거야?"
            },
            {
                "speaker": "알 수 없음",
                "text": " 어떤"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 트리퍼"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 와 이거"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 정상적으로 지금 워킹하지 않는다"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 아이딘"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 하하하하"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 네"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 이게 수동으로 밑에"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 쟤가 약간 색상 다르게 좀"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 스크립트 단에다가"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 해주면 좋을 것 같아"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그냥 아예"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 나중에 뭐"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 안 되면 되잖아"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그냥 순수 바닐라"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 거기 있는 거에"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 자바 스크립트로"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 제목하고 있는"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 작성을 하면"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 이렇게"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 사실 이런 데서는"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그렇게 하면"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 아직 기출을 안 한 것 같은데"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그렇게 하면"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 저 바꿔놨는데"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 워킹은 해야 돼"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그렇게 되면"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 얘랑"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 얘 방금 안 한다고"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 다이나믹형이 아니라"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 색상만 다르게"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 그냥 하드코딩해서"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 저래고"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 이거는 나중에 잡으면 되지"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 갖고 가면 이슈 없다는 거죠"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 아 그게"
            },
            {
                "speaker": "SPEAKER_01",
                "text": " 어디 색상이야?"
            }
        ]
    }
    send_minutes(meeting_id, content, b_task)

def make_summary(meeting_id: int, content: dict):
    print("make_summary")
    summary = sLLM.sllm_response(content)

    send_summary(meeting_id, summary)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)