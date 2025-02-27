import os
import sys
from model import process_photo_feedback, generate_image_prompt, flux_generate_image, answer_feedback_questions, update_photo_feedback

def main():
    print("📷 사진 피드백 시스템에 오신 것을 환영합니다!")

    # 사용자 입력 받기
    image_url = input("이미지 URL을 입력하세요: ")
    user_preference = input("원하는 사진 스타일이나 개선점에 대한 요청을 입력하세요: ")

    # 초기 사진 피드백 생성
    feedback = process_photo_feedback(image_url, user_preference)
    print("\n📷 초기 피드백:")
    print(feedback)

    # 피드백 기반 예시 이미지 생성
    print("\n🖼️ 피드백을 반영한 예시 이미지 생성 중...")
    image_prompt = generate_image_prompt(image_url, feedback, user_preference)  # 세밀한 이미지 프롬프트 생성
    flux_generate_image(image_prompt)  # 이미지 생성 함수 호출
    print("✅ 예시 이미지가 'output.jpg'로 저장되었습니다.")

    print("\n🔎 생성된 이미지 프롬프트:")
    print(image_prompt) # 확인용 

    while True:
        print("\n🔹 옵션을 선택하세요:")
        print("1. 피드백에 대한 추가 질문")
        print("2. 새로운 사진 업로드 및 업데이트된 피드백 받기")
        print("3. 종료")

        choice = input("입력: ")

        if choice == "1":
            user_question = input("💬 추가로 궁금한 점을 입력하세요: ")
            answer = answer_feedback_questions(feedback, user_question)
            print("\n💡 추가 질문 응답:")
            print(answer)

        elif choice == "2":
            new_image_url = input("📷 새로운 이미지 URL을 입력하세요: ")
            updated_feedback = update_photo_feedback(new_image_url, user_preference, feedback)
            feedback = updated_feedback  # 업데이트된 피드백 저장
            print("\n✅ 업데이트된 피드백:")
            print(updated_feedback)

        elif choice == "3":
            print("📴 프로그램을 종료합니다. 감사합니다!")
            sys.exit()

        else:
            print("⚠️ 올바른 옵션을 선택하세요!")

if __name__ == "__main__":
    main()
