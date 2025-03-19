from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationSummaryMemory
from dotenv import load_dotenv
from supabase import create_client, Client
from datetime import datetime

load_dotenv()

# Supabase 클라이언트 초기화
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# 상담 챗봇 
# 모델 및 프롬프트 설정
chat_model = ChatOpenAI(temperature=0.7, model="ft:gpt-4o-mini-2024-07-18:personal::AqCIAN5W") # temperature를 0.0으로 변경
memory = ConversationSummaryMemory(llm=chat_model, return_messages=False)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
     <role>
    As a motivational interviewing (MI) expert, your role is to empathetically understand the client’s situation, identify concerns, and guide them toward solutions. Focus on their ambivalence, helping them recognize the need for change, discover positive motivation, and initiate voluntary behavioral changes. 

    Use a warm and empathetic tone, listening closely to their issues, and highlighting their strengths to foster motivation. Once their concerns are clear, offer practical strategies or specific actions to address the issues. Clarify the gap between their current state and desired goals to help them recognize and feel the need for change.
     </role>
      
Include MITI Behavior Codes: At the start of each of your responses, place one or more MITI Behavior Codes in parentheses. Select the code that best aligns with your primary goal in each response. You may use multiple codes if applicable. Choose from the following list: 
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
session_id = 250100000

def chat_with_bot_and_store(user_input):
    # 1. 메시지 생성 시간
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 2. 챗봇 응답 생성
    history =  memory.buffer
    input_message = prompt.format_messages(user_message=user_input, history=history)
    response = chat_model.invoke(input_message)
    memory.save_context({"input": user_input}, {"output": response.content})
    bot_response = response.content
    
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