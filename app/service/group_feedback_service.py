import re
from supabase import create_client, Client
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from app.database import get_db

# 환경 변수 로드
load_dotenv()

# Supabase 클라이언트 초기화
db = get_db()

#세션별 개선사항 수집
def get_all_feedbacks(session_id_list: list) -> list:
    all_feedbacks = []

    for session_id in session_id_list:
        response = db.table("session").select("feedback").eq("session_id", session_id).eq("is_closed", True).execute()

        if response.data:
            for item in response.data:
                feedback_text = item.get("feedback")

                if feedback_text:
                # 이스케이프 문자 해제
                    feedback_text = feedback_text.replace("\\n", "\n").replace("\\r", "\r")
                
                if feedback_text:
                    # 정규 표현식을 사용하여 개선 사항 목록 추출
                    match = re.search(r"\[개선사항\]\s*(•.*)", feedback_text, re.DOTALL)
                    if match:
                        feedbacks = match.group(1).strip().split("•")
                        for imp in feedbacks:
                            imp = imp.strip()
                            if imp: # 빈 문자열이 아닌 경우에만 추가
                                all_feedbacks.append(imp)
                    else:
                        print(f"No feedbacks section found for session_id: {session_id}")
                else:
                    print(f"No feedbacks data found for session_id: {session_id}")
        else:
            print(f"No data found for session_id: {session_id}")

    return all_feedbacks


# 부서별 개선사항 요약 모델 
def summarize_feedbacks(text: list) -> str:
    try:
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7) # GPT-4 사용
        prompt = PromptTemplate(
            input_variables=["text"],
            template=
            """
            <role>
            ""Please organize the following list by removing duplicate or similar items. Combine similar ones into a single entry and categorize them under relevant topics. Keep the summary concise and clear. Remove satisfaction ('만족하고 있음').":\n{text}"
            you use only korean.
            
            If there are no improvements print the following:
            • 만족하고 있음
            
            Please use only korean. Short answer only. 
            </role>
            
            <format>
            Folloing the format:
            [subject]
             - topic
             - topic
            [subject]
             - topic
             - topic
            </format>

            """ 
        )
        chain = prompt | llm

        # 입력 텍스트를 문자열로 변환
        text_string = " ".join(text) # 문장들을 공백으로 연결
        summary = chain.invoke(text_string)
        
        return summary.content # content 속성 접근
    except Exception as e:
        print(f"Error during summarization: {e}")
        return "요약 중 오류가 발생했습니다."

# 파이프 라인    
def feedback_pipeline(session_id_list: list[str]) -> str:
    
    all_feedbacks = get_all_feedbacks(session_id_list)

    if not all_feedbacks:
        return "[개선사항]\n• 만족하고 있음"

    summary = summarize_feedbacks(all_feedbacks)

    return summary
