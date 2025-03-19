from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
from langchain.memory import ConversationSummaryMemory
from dotenv import load_dotenv
from supabase import create_client, Client
from datetime import datetime
import re

load_dotenv()

# Supabase 클라이언트 초기화
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# 상담 챗봇 
# 모델 및 프롬프트 설정
chat_model = ChatOpenAI(temperature=0.7, model="ft:gpt-4o-mini-2024-07-18:personal::Aprtgs0u") # temperature를 0.0으로 변경
memory = ConversationSummaryMemory(llm=chat_model, return_messages=False)
prompt = ChatPromptTemplate.from_messages([
    ("system", """       
     <role>
     You are an expert in motivational interviewing. Your role is to understand the client’s situation, identify the problem, and propose solutions. During this process, you must focus on addressing the client's ambivalent feelings. For example, a client might want to improve their relationship with a coworker without reconciling with them.
     Please conduct the interview only in Korean.
     </role>
     
     Conduct the counseling session using the following interview techniques
      - Include MITI Behavior Codes: At the start of each of your responses, place one or more MITI Behavior Codes in parentheses. Select the code that best aligns with your primary goal in each response. You may use multiple codes if applicable. Choose from the following list: 
      
     <code>
     Complex Reflection - (CR) 
     Simple Reflection - (SR)
     Affirm - (AF)
     Emphasizing Autonomy - (Emphasize)
     Seeking Collaboration - (Seek)
     Question - (Q) 
     Giving Information - (GI)
     Persuade with Permission - (Persuasion with)
     Persuade - (Persuasion)
     Confront - (Confront)
     </code>
     

        """),
    ("user", "{user_message}"),
    ("ai", "{history}"),
])


# 세션 ID 받아오기(지금은 예시로 고정값)
session_id = 250115001


def clean_response(response):
    """
    챗봇 응답에서 (SR), (AF) 등의 태그를 제거하는 함수.
    """
    pattern = r"\((?:\s*SR|CR|AF|Emphasize|Seek|Q|GI|Persuasion with|Persuasion|Confront|Simple Reflection|Affirm|Emphasizing Autonomy|Seeking Collaboration|Question|Giving Information|Persuade with Permission|Persuade\s*)(?:,\s*(?:SR|CR|AF|Emphasize|Seek|Q|GI|Persuasion with|Persuasion|Confront|Simple Reflection|Affirm|Emphasizing Autonomy|Seeking Collaboration|Question|Giving Information|Persuade with Permission|Persuade))*\)"
    # 태그 제거
    cleaned_response = re.sub(pattern, "", response)
    # 불필요한 공백 제거
    return cleaned_response.strip()

def chat_with_bot_and_store(user_input):
    # 1. 메시지 생성 시간
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 2. 챗봇 응답 생성
    history =  memory.buffer
    input_message = prompt.format_messages(user_message=user_input, history=history)
    response = chat_model.invoke(input_message)
    memory.save_context({"input": user_input}, {"output": response.content})
    bot_response = clean_response(response.content)
    
    # 3. 데이터베이스에 사용자 메시지 저장
    supabase.table("Message").insert({
        "session_id": session_id,
        "content": user_input,
        "is_question": True,
        "created_at": created_at
    }).execute()
    
    # 4. 데이터베이스에 챗봇 응답 저장
    supabase.table("Message").insert({
        "session_id": session_id,
        "content": bot_response,
        "is_question": False,
        "created_at": created_at
    }).execute()
    
    return bot_response

# 대화 루프
if __name__ == "__main__":
    print("감정본부: 안녕하세요! 감정본부입니다. 당신의 감정은 어떠한가요?")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["종료"]:
            print("감정본부: 대화를 종료합니다. 좋은 하루 되세요!")
            break
        else:
            response = chat_with_bot_and_store(user_input)
            print(f"Chatbot: {response}")