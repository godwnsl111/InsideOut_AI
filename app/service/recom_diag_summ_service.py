from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from app.database import get_db
from app.config import OPENAI_API_KEY

# OpenAI 모델 초기화
chat_model = ChatOpenAI(
    temperature=0.7,
    model="gpt-4o",
    api_key=OPENAI_API_KEY,
    max_tokens=1000
)

# db 연결
db = get_db()

# session 상태 확인 함수(True, False)
def get_is_closed(session_id: int) -> bool:
    try: 
        response = db.table("session").select("is_closed").eq("session_id", session_id).execute()
        if response.data and isinstance(response.data[0].get("is_closed"), bool):
            return response.data[0]["is_closed"]
        elif response.error:
            raise Exception(f"Error: {response.error}")
    except Exception as e:
        raise Exception(f"Error checking session status: {e}")

# message 테이블에서 content 가져오기
def get_user_messages_and_ors(session_id: int):
    try:
        # 메시지 가져오기
        response = db.table("message").select("content").eq("session_id", session_id).eq("author_type", "USER").execute()
        messages = [msg["content"] for msg in response.data]
        # ORS 점수 가져오기
        ors_response = db.table("session").select("ors_score").eq("session_id", session_id).single().execute()
        ors_score = ors_response.data["ors_score"]
    
        return messages, ors_score
    
    except Exception as e:
        raise Exception(f"Error fetching user data: {e}")




# 전문 상담가 필요 여부 판단
def needs_professional_help(ors_score, messages):
    try:
        # 기준 1: ORS 점수 기준
        if ors_score <= 21:
            return True, "RISK"
        elif ors_score > 21:
            # 기준 2: 메시지 감정 분석
            negative_keywords = ["힘들다", "포기", "죽고 싶다", "불안", "우울", "외롭다"]
            negative_count = sum(1 for msg in messages if any(kw in msg for kw in negative_keywords))
            if negative_count / len(messages) > 0.4:  # 부정적 메시지가 40% 이상
                return True, "RISK"
    
        return False, "STABLE"
    
    except Exception as e:
        raise Exception(f"Error analyzing professional help need: {e}")


def summary_counseling(ors_score, messages):
    try:
        messages_text = "\n".join(messages)

        # 프롬프트 작성
        d_prompt = f"""
        You are a counselor analyzing psychological sessions. ors_score = {ors_score}
        [요약] Based on the following {messages_text} and ors_score, analyze the user's mental and emotional state: Provide a concise analysis of the user's state and include empathy and advice to address their situation. A situation can be considered risky if the ors_score is 21 or below.
        [제안] Provide actionable suggestions in a paragraph format to help the user improve their situation or make positive changes.

        Write the counseling result in Korean, and ensure each section is clearly labeled as [요약] and [제안].
        <format>
        [요약]
        paragraph
        [제안]
        paragraph
        </format>
        """
        
        response = chat_model.invoke([
            {"role": "system", "content": "You are a psychologist."},
            {"role": "user", "content": d_prompt}
        ])
        
        return response.content
    
    except Exception as e:
        raise Exception(f"Error analyzing user state: {e}")


# 개선사항 도출
def feedback(messages):
    try:
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7) # GPT-4 사용
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            <role>
            ""As a consulting expert, identify potential workplace improvements based on the consultation details, ensuring the extracted improvements are precise and concise.":\n{text}"
            Phrase improvements positively
            Exclude anything unrelated to the company.
            If there are no improvements related to the company, print the following:
            "[개선사항]
            • 만족하고 있음"
            Please use only korean. Short answer only.
            </role>
            
            <format>
            Folloing the format:
            "[개선사항]
            • Enhanced departmental communication
            • Better work distribution"
            </format>

            """
            
        )
        chain = prompt | llm

        # 입력 텍스트를 문자열로 변환
        text_string = " ".join(messages) # 문장들을 공백으로 연결
        feedback = chain.invoke(text_string)
        
        return feedback.content # content 속성 접근
    
    except Exception as e:
        print(f"Error during summarization: {e}")
        return "요약 중 오류가 발생했습니다."



def diagnosis(session_id: int):
    # 메시지와 ORS 점수 가져오기
    messages, ors_score = get_user_messages_and_ors(session_id)
    
    # 전문가 도움 여부
    status = needs_professional_help(ors_score, messages)
    
    # 상담 내용 요약
    summary = summary_counseling(ors_score, messages)
    
    # 개선 사항 요약
    feedback_total = feedback(messages)
    
    return {
            "status": status[1],
            "summary": summary,
            "feedback": feedback_total
        }
