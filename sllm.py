import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

class SLLM:
    def __init__(self):
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, # 모델을 4비트 정밀도로 로드
            bnb_4bit_quant_type="nf4", # 4비트 NormalFloat 양자화: 양자화된 파라미터의 분포 범위를 정규분포 내로 억제하여 정밀도 저하 방지
            bnb_4bit_use_double_quant=True, # 이중 양자화: 양자화를 적용하는 정수에 대해서도 양자화 적용
            bnb_4bit_compute_dtype=torch.bfloat16 # 연산 속도를 높이기 위해 사용 (default: torch.float32)
        )
        self.model_path = './summary-finetuned-aya'
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=self.bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model_max_length = 128000
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, max_length=self.model_max_length)

    def get_prompt(self):
        prompt = """아래 지시사항에 따라 사용자가 입력하는 회의록을 요약하십시오.

        회의록의 **주요 논의 주제**를 포괄적으로 요약하십시오.
        요약은 아래와 같은 트리구조로 작성하십시오.
        1. **회의 주제**: [주제]
        2. **회의 요약**:
        - [요약 1]
        - [요약 2]
        - [요약 3]
        ...
        3. **회의 결론**: [결론]

        - 반드시 문어체를 사용하십시오.
        - 요약문은 한국어로 작성하십시오.
        - 회의록에 없는 내용은 입력하지 마십시오.
        - 모든 논의된 주제를 빠짐없이 포함하십시오.
        - 회의의 주요 내용을 회의 요약에 모두 포함하십시오.
        - 가능하면 논의된 각 주제의 맥락을 충분히 설명하십시오.
        - 여러 팀이 참여한 경우, 팀별로 나누어 요약을 작성하십시오.
        - 다음 회의 날짜가 명시된 경우에만 다음 회의 일정을 별도로 표시하십시오.
        - 다음 회의에 관한 언급이 없을 경우 다음 회의 일정을 표시하지 마십시오.
        - 중복된 내용은 제거하고, 각 논의 주제를 명확히 구분하여 작성하십시오.

        **요약문은 반드시 한글로 작성하십시오.**
        """

        return prompt
    
    def get_minutes(self, minutes:dict):
        minutes_arr = minutes.get('minutes')
        new_minutes = []

        for line in minutes_arr:
            speaker = line.get("speaker").strip()
            text = line.get("text").strip()
            content = f"{speaker}: {text}"

            new_minutes.append(content)
        
        minutes_str = '\n'.join(new_minutes)
        return minutes_str

    
    def make_message(self, minutes:dict):

        prompt = self.get_prompt()
        minutes = self.get_minutes(minutes)

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": minutes},
        ]

        message = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        
        return message
    
    def make_format(self, response: str):
        start_idx = response.find('1.')
        return response[start_idx:]

    def sllm_response(self, minutes: dict):
        # 텍스트 생성을 위한 파이프라인 설정
        message = self.make_message(minutes)
        
        outputs = self.pipe(
            message,
            do_sample=True,
            temperature=0.4,
            top_k=5,
            top_p=0.8,
            repetition_penalty=1.2,
            add_special_tokens=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        response = self.make_format(outputs[0]["generated_text"][len(message):])
        return response