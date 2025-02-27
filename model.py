import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

import replicate

# .env 파일 로드
load_dotenv()

# 환경 변수에서 OPENAI API 키 가져오기
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("환경 변수 OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

# GPT-4o 모델 초기화
llm = ChatOpenAI(model="gpt-4o", temperature=0.7, openai_api_key=openai_api_key)


# 1-1. 초기 사진 피드백 생성
def process_photo_feedback(image_url, user_preference):
    """
    사용자의 요청을 반영하여 촬영 피드백을 생성하는 함수.
    """
    prompt = f"""
        인물 사진을 분석하고 피드백을 제공합니다. 사용자가 '{user_preference}'라고 요청했습니다. 
        우선, 현재 사진이 어떤지, 잘 찍었는지 간단히 판단해주세요.
        구도 조정, 포즈&표정, 카메라 앵글, 광원의 위치, 줌인·줌아웃 등의 관점에서 개선이 필요한 점이 있는 요소에 대해서만 설명해주고, 개선할 필요가 없는 요소는 생략해주세요.
    """
    
    messages = [
        SystemMessage(content="당신은 사진 전문가입니다. 사용자에게 친절한 말투로 알려주세요."),
        HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}}
        ])
    ]
    
    response = llm(messages)
    return response.content


# 1-2. 예시 이미지 생성 함수 
def generate_image_prompt(image_url, feedback, user_preference):
    """
    Generates a detailed prompt for an image generation model based on the feedback and user preference.
    """
    prompt = f"""
        Describe the size of the image and the size of the subject within it.
        Provide a highly detailed description of the subject’s clothing, including the top, bottom, and shoes, specifying color, material, and fit.
        Describe the weather and background in great detail.
        Give a comprehensive description of the entire scene so that even someone who has never seen the photo can fully visualize it.

        Based on {feedback}, describe the best camera angle and composition for the subject, such as head shot, upper body shot, half body shot, full body shot, bird’s eye shot, drone shot, high angle shot, from above, knee shot, etc.
        Specify the exact placement and proportion of the subject within the rule of thirds, providing precise measurements (e.g., lower center of the frame, occupying 1/2 of the image in a smaller composition).

        To satisfy {user_preference}, describe in detail the ideal facial expression and pose the subject should take.

        Write a highly detailed description of the entire scene within approximately 1000 characters so that anyone can accurately visualize it.
        """
    
    messages = [
        SystemMessage(content="You are someone who describes photographs. Describe the image so vividly and precisely that even someone who has not seen it can perfectly visualize it in their mind."),
        HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}}
        ])
    ]
    
    response = llm(messages)
    return response.content

# 1-3. flux 모델을 활용한 이미지 생성
def flux_generate_image(input_prompt):

    input = {
        "prompt": f'{input_prompt}\n 위 내용을 기반으로 3:4 비율의 이미지를 생성해줘.',
        "aspect_ratio": "3:4",
        "prompt_upsampling": True
    }

    output = replicate.run(
        "black-forest-labs/flux-1.1-pro",
        input=input
    )

    with open("output.jpg", "wb") as file:
        file.write(output.read())

# 2. 추가 질문 처리
def answer_feedback_questions(feedback, user_question):
    """
    사용자가 피드백에 대해 추가 질문을 할 경우 응답을 생성하는 함수.
    """
    prompt = f"""
        사용자가 '{user_question}'라고 추가 요청했습니다. \n
        {feedback}
        사용자의 질문에 대한 답변을 해주세요.
        이전에 받은 피드백과 사용자의 추가 요청을 반영하여, 현재 사진에서 어떻게 수정해야할지 구체적으로 알려주세요.\n
    """
    
    messages = [
        SystemMessage(content="당신은 사진 전문가입니다. 사용자에게 친절한 말투로 알려주세요."),
        HumanMessage(content=prompt)
    ]
    
    response = llm(messages)
    return response.content


# 3. 새로운 사진 평가 및 피드백 업데이트
def update_photo_feedback(new_image_url, user_preference, old_feedback):
    """
    새로운 사진을 평가하여 기존 피드백을 반영한 추가 피드백을 생성.
    """
    prompt = f"""
    사용자가 새로운 사진을 업로드했습니다.
    기존 피드백: {old_feedback}
    새로운 사진이 피드백을 잘 반영했다면 칭찬의 글을 남겨주세요. 백점 만점의 몇 점인지 알려주세요.
    추가적으로 수정이 필요한 부분이 있다면 알려주세요.
    """
    
    messages = [
        SystemMessage(content="당신은 사진 전문가입니다. 사용자에게 친절한 말투로 알려주세요."),
        HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": new_image_url}}
        ])
    ]
    
    response = llm(messages)
    return response.content
