import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import openai

# .env 파일 로드
load_dotenv()

# 환경 변수에서 OPENAI API 키 가져오기
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("환경 변수 OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

# GPT-4o 모델 초기화
llm = ChatOpenAI(model="gpt-4o", temperature=0.7, openai_api_key=openai_api_key)

POSE_OPTIONS = [
    "head", "upper body", "half body", "full body", "bird eye shot", "drone shot", "high angle shot",
    "from below", "from above", "knee shot", "medium shot", "perspective, full body", "perspective, upper body",
    "dutch angle", "bust shot"
]

# 금지된 단어 리스트 (예제)
FORBIDDEN_WORDS = ["nudity", "violence", "weapon", "blood", "gore"]

def sanitize_prompt(prompt):
    """
    프롬프트에서 금지된 단어를 필터링하는 함수.
    """
    for word in FORBIDDEN_WORDS:
        prompt = prompt.replace(word, "[filtered]")
    return prompt

def recommend_pose_options(feedback):
    """
    GPT 피드백을 바탕으로 적절한 촬영 구도를 추천하는 함수.
    """
    prompt = f"""
    다음 피드백을 바탕으로 적절한 촬영 구도를 추천하세요:
    {feedback}
    추천할 때 반드시 다음 리스트에서 선택해야 합니다: {', '.join(POSE_OPTIONS)}
    최대 3개만 추천하세요.
    """
    
    messages = [
        SystemMessage(content="당신은 사진 전문가입니다."),
        HumanMessage(content=prompt)
    ]
    
    response = llm(messages)
    return response.content.split(", ")[:3]  # 최대 3개만 반환

def process_photo_feedback(image_url, user_preference):
    """
    사용자의 요청을 반영하여 촬영 피드백을 생성하는 함수.
    """
    prompt = f"""
    인물 사진을 분석하고 피드백을 제공합니다. 사용자가 '{user_preference}'라고 요청했습니다. 
    현재 사진이 어떤지 판단하고, 구도, 포즈, 카메라 앵글, 광원 등에 대해 개선이 필요한 점을 설명하세요.
    """
    
    messages = [
        SystemMessage(content="당신은 사진 전문가입니다."),
        HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}}
        ])
    ]
    
    response = llm(messages)
    return response.content

def generate_dalle_prompt(feedback):
    """
    GPT의 피드백을 기반으로 DALL·E 이미지 생성 프롬프트를 생성하는 함수.
    """
    pose_suggestions = recommend_pose_options(feedback)
    pose_text = ", ".join(pose_suggestions)
    
    prompt_template = """
    다음 설명을 기반으로 새로운 사진을 생성하세요:
    {feedback}
    추천된 촬영 구도: {pose_text}
    라인 아트 스타일로 표현하세요. 간단하고 선명한 선을 사용하여 깔끔한 스케치 스타일을 유지하세요.
    """
    
    sanitized_prompt = sanitize_prompt(prompt_template.format(feedback=feedback, pose_text=pose_text))
    return sanitized_prompt

def generate_image_from_feedback(feedback):
    """
    GPT의 피드백을 기반으로 OpenAI DALL·E API를 호출하여 이미지를 생성하는 함수.
    """
    dalle_prompt = generate_dalle_prompt(feedback)
    response = openai.images.generate(
        model="dall-e-3",
        prompt=dalle_prompt,
        n=1,
        size="1024x1024"
    )
    return response.data[0].url

def run_photo_feedback_system():
    """
    사용자의 입력을 받아서 사진 피드백을 제공하고, 피드백을 기반으로 이미지를 생성하는 흐름을 실행.
    """
    print("사진 피드백 시스템을 시작합니다.\n")
    image_url = input("사진의 URL을 입력하세요: ")
    user_preference = input("어떤 점을 개선하고 싶은지 입력하세요: ")
    
    feedback = process_photo_feedback(image_url, user_preference)
    print("\n초기 피드백:\n", feedback)
    
    try:
        generated_image = generate_image_from_feedback(feedback)
        print("\n생성된 이미지 URL:\n", generated_image)
    except openai.BadRequestError as e:
        print("\n[ERROR] 이미지 생성이 차단되었습니다. 콘텐츠 정책을 위반할 가능성이 있는 내용을 확인해주세요.")
        print("에러 메시지:", str(e))

if __name__ == "__main__":
    run_photo_feedback_system()
