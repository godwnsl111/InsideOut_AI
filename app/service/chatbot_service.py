import re
import pytz
import chromadb
import base64
import requests

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationSummaryMemory, CombinedMemory, ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

from datetime import datetime
from app.config import OPENAI_API_KEY
from app.database import get_db
from openai import OpenAI

# db 연결
db = get_db()

# Chroma 초기화
client = chromadb.Client()
collection = client.create_collection("chatbot_data")

# OpenAI 모델 초기화
chat_model = ChatOpenAI(
    temperature=0.7,
    # model="ft:gpt-4o-mini-2024-07-18:personal::AqVCT2m3", # 100 / 5 / 1
    # model="ft:gpt-4o-mini-2024-07-18:personal::Ay7vsdbd", # 100 / 1 / 1
    # model="ft:gpt-4o-mini-2024-07-18:personal::Ay8La02V", # 10 / 1 / 1
    model="ft:gpt-4o-mini-2024-07-18:personal::Ay8PqjEF", # 10 / 3 / 1 >> 후보
    
    # model="gpt-4o",
    # model="ft:gpt-4o-2024-08-06:personal::AyBiLXwZ", # 10 / 3 / 1
    # model="ft:gpt-4o-2024-08-06:personal::AyDlgTPl", # 10 / 1 / 1
    # model="ft:gpt-4o-2024-08-06:personal::AyDoZdFh", # 10 / 2 / 1
    api_key=OPENAI_API_KEY,
    max_tokens=1000,
    frequency_penalty=0.7  # 0에서 2 사이의 값. 높을수록 반복을 더 강하게 억제합니다.
)

# OpenAI 클라이언트 초기화
vision_client = OpenAI(api_key=OPENAI_API_KEY)

# 전역 변수로 vectorstore 선언
vectorstore = None

# session_id 로 해당 유저의 다른 세션을 포함한 모든 메시지 검색
def fetch_user_messages_by_session(session_id):
    try:
        # 1. 주어진 session_id로 user_id 가져오기
        session_response = db.table("session").select("user_id").eq("session_id", session_id).single().execute()

        user_id = session_response.data.get("user_id")
        if not user_id:
            print(f"session_id {session_id}에 해당하는 user_id를 찾을 수 없습니다.")
            return []

        # 2. user_id로 해당 유저의 모든 session_id 검색
        sessions_response = db.table("session").select("session_id").eq("user_id", user_id).execute()

        session_ids = [record["session_id"] for record in sessions_response.data]
        if not session_ids:
            print(f"user_id {user_id}가 가진 세션이 없습니다.")
            return []

        # 3. session_id 목록으로 message 테이블에서 데이터 가져오기
        messages_response = db.table("message")\
            .select("message_id, session_id, content, author_type, created_at").in_("session_id", session_ids).eq("author_type", "USER").execute()

        messages = messages_response.data
        if not messages:
            print(f"session_ids {session_ids}에 해당하는 메시지가 없습니다.")
            return []

        # 4. 데이터를 Document 형식으로 변환
        documents = []
        for message in messages:
            content = message.get("content", "")
            metadata = {
                "message_id": message.get("message_id"),
                "session_id": message.get("session_id"),
                "created_at": message.get("created_at"),
                "author_type": message.get("author_type"),
            }
            documents.append(Document(page_content=content, metadata=metadata))

        return documents

    except Exception as e:
        print(f"오류 발생: {e}")
        return []

# OpenAI 임베딩을 사용하여 텍스트를 벡터로 변환
embeddings = OpenAIEmbeddings()

# Chroma에 데이터를 저장하는 함수
def initialize_chroma_with_supabase_data(session_id):
    try:
        documents = fetch_user_messages_by_session(session_id)
        if not documents:
            print("Chroma 초기화를 위한 문서가 없습니다.")
            return

        # 메타데이터에서 복잡한 타입 필터링
        for doc in documents:
            # created_at을 문자열로 변환
            if doc.metadata.get('created_at'):
                doc.metadata['created_at'] = str(doc.metadata['created_at'])
            # None 값을 가진 메타데이터 제거
            doc.metadata = {k: v for k, v in doc.metadata.items() if v is not None}

        global vectorstore
        vectorstore = Chroma.from_documents(documents, embeddings, collection_name="chatbot_data")
        print(f"Chroma 데이터 초기화 완료: {len(documents)}개의 문서가 삽입되었습니다.")

    except Exception as e:
        print(f"Chroma 초기화 중 오류 발생: {e}")

# 질문에 대한 임베딩 생성
query = "사용자 질문 내용"

# 세션별 메모리 풀
session_memory_pool = {}


def clean_response(response: str) -> str:
    """
    GPT 응답에서 불필요한 태그 제거
    """
    pattern = r"\((?:\s*SR|CR|AF|Emphasize|Seek|Q|GI|Persuasion with|Persuasion|Confront)\)"
    return re.sub(pattern, "", response).strip()


def get_session_memory(session_id: int):
    """
    세션 ID에 해당하는 메모리를 반환.
    없으면 새로운 메모리를 생성.
    """
    global session_memory_pool
    
    if session_id not in session_memory_pool:
        # 최근 대화를 저장하는 버퍼 메모리
        buffer_memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="input",
            output_key="output",
            return_messages=True,
            k=5  # 최근 5개 대화 저장
        )
        
        # 전체 대화의 요약을 저장하는 메모리
        summary_memory = ConversationSummaryMemory(
            llm=chat_model,
            input_key="input",
            output_key="output",
            memory_key="conversation_summary",
            return_messages=False
        )
        
        # 두 메모리 결합
        session_memory_pool[session_id] = CombinedMemory(
            memories=[buffer_memory, summary_memory],
            input_key="input",
            output_key="output"
        )
    
    return session_id, session_memory_pool[session_id]


def get_latest_message(session_id: int) -> dict:
    """
    세션 ID에 해당하는 가장 최근 메시지와 이미지 URL을 조회
    """
    try:
        response = (db.table("message")
                   .select("content, image_url")
                   .eq("session_id", session_id)
                   .order("created_at", desc=True)
                   .limit(1)
                   .execute())
        messages = response.data
        if not messages:
            print(f"Session {session_id}의 메시지를 찾을 수 없습니다.")
            return {"content": "", "image_url": None}
        return {
            "content": messages[0].get('content', ''),
            "image_url": messages[0].get('image_url')
        }
    except Exception as e:
        print(f"Error fetching message: {str(e)}")
        return {"content": "", "image_url": None}


async def get_session_ors_scores(user_id: int):
    """현재 및 이전 세션의 ORS 점수를 조회합니다."""
    try:
        # 최근 2개의 세션을 생성일자 기준 내림차순으로 가져오기
        response = (db.table("session").select("ors_score").eq("user_id", user_id).order("created_at", desc=True).limit(2).execute())

        sessions = response.data  # Supabase 응답 데이터

        if not sessions:
            return None, None

        # 최근 세션의 ORS 점수 할당
        current_score = sessions[0]["ors_score"] if len(sessions) > 0 else None
        previous_score = sessions[1]["ors_score"] if len(sessions) > 1 else None

        return current_score, previous_score

    except Exception as e:
        print(f"ORS 점수 조회 중 오류 발생: {str(e)}")
        return None, None


async def get_image_description(image_url: str) -> str:
    """
    이미지 URL을 받아서 GPT-4 Vision을 통해 이미지 설명을 생성
    """
    try:
        # 이미지 URL에서 이미지 다운로드
        response = requests.get(image_url)
        if response.status_code != 200:
            print(f"이미지 다운로드 실패: {response.status_code}")
            return ""

        # 이미지를 base64로 인코딩
        image_base64 = base64.b64encode(response.content).decode('utf-8')
        
        # MIME 타입 결정 (간단한 버전)
        content_type = response.headers.get('content-type', 'image/jpeg')
        
        # base64 URL 형식으로 변환
        base64_url = f"data:{content_type};base64,{image_base64}"

        response = vision_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "이 이미지에 대해 자세히 설명해주세요."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": base64_url
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"이미지 설명 생성 중 오류 발생: {str(e)}")
        return ""


async def chat_with_bot(session_id: int, user_input: str):
    try:
        # Vectorstore 초기화 여부 확인
        if not vectorstore:
            print("Vectorstore가 초기화되지 않았습니다.")
            return {
                "content": "DB 초기화 오류가 발생했습니다. 관리자에게 문의하세요.",
                "authorType": "AI"
            }

        # 사용자 입력으로 벡터 검색
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(user_input)  # 관련 문서 검색

        # 검색된 문서 내용 병합
        relevant_context = "\n".join([doc.page_content for doc in relevant_docs])
        
        # 기본 프롬프트 (시스템 메시지)
        default_prompt = """       
             <role>
     As a motivational interviewing (MI) expert, your role is to empathetically understand the client's situation, identify concerns, and guide them toward solutions. Focus on their ambivalence, helping them recognize the need for change, discover positive motivation, and initiate voluntary behavioral changes. Clarify the discrepancies between the client's current state and their desired goals to promote change.
     Please conduct the interview only in Korean.

     Use a warm and empathetic tone, listening closely to their issues, and highlighting their strengths to foster motivation. Once their concerns are clear, offer practical strategies or specific actions to address the issues. Clarify the gap between their current state and desired goals to help them recognize and feel the need for change.
     </role>
    
     <context>
     Here is some relevant information retrieved from past conversations or knowledge base:
     {}
     </context>

     Below are the MITI Behavior Codes. Select the code that best aligns with your primary goal in each response. You may use multiple codes if applicable.

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
        You should appropriately use Simple Reflection (SR) to repeat or paraphrase the client's statements and Complex Reflection (CR) to reflect the underlying emotions and meanings in their words.
        
        Be cautious not to overuse Affirm (AF).
        """.format(relevant_context)  # 여기서 relevant_context를 직접 포매팅
        
        try:
            # 현재 세션 정보 조회 (Supabase 방식)
            response = db.table("session").select("user_id").eq("session_id", session_id).single().execute()
            
            if not response.data:
                return await generate_chat_response(session_id, user_input, default_prompt)

            user_id = response.data.get("user_id")  # 유저 ID 가져오기

            # 현재와 이전 세션의 ORS 점수 조회
            current_score, previous_score = await get_session_ors_scores(user_id)

            # ORS 점수가 없는 경우 기본 프롬프트 사용
            if current_score is None:
                return await generate_chat_response(session_id, user_input, default_prompt)

            
            # ORS 점수에 따른 프롬프트 선택
            if current_score >= 25:
                base_strategy = """
                <strategy>
                Focus on goal-setting and action planning. Help the client:
                1. Set specific, achievable goals
                2. Identify their strengths and resources
                3. Develop concrete action steps
                4. Create a timeline for implementation
                5. Plan for potential obstacles
                </strategy>
                """
            else:
                base_strategy = """
                <strategy>
                Focus on emotional support and reflection. Prioritize:
                1. Deep emotional understanding
                2. Complex reflections of feelings
                3. Validation of experiences
                4. Creating a safe space
                5. Building therapeutic alliance
                </strategy>
                """
            
            # ORS 점수 변화에 따른 추가 지시사항
            if previous_score is not None:
                if current_score > previous_score:
                    score_change_prompt = """
                    <additional_focus>
                    The client's ORS score has improved. During the conversation, find an appropriate moment to:
                    1. Explore what positive changes have occurred
                    2. Ask what they did differently that led to improvement
                    3. Reinforce their successful strategies
                    4. Help them identify how to maintain this progress
                    </additional_focus>
                    """
                elif current_score < previous_score:
                    score_change_prompt = """
                    <additional_focus>
                    The client's ORS score has decreased. During the conversation:
                    1. Gently explore what challenges they've faced recently
                    2. Ask about what has changed since the last session
                    3. Show empathy and understanding for their struggles
                    4. Help identify what support they need right now
                    </additional_focus>
                    """
                else:
                    score_change_prompt = """
                    <additional_focus>
                    The client's ORS score remains the same. During the conversation:
                    1. Explore their current situation and any changes
                    2. Identify what's working well and what isn't
                    3. Discuss what might help create positive movement
                    </additional_focus>
                    """
            else:
                score_change_prompt = ""
            
            # 최종 프롬프트 조합
            custom_prompt = default_prompt + base_strategy + score_change_prompt
            
            # 여기를 수정: 프롬프트를 직접 반환하지 않고 generate_chat_response 호출
            return await generate_chat_response(session_id, user_input, custom_prompt)
            
        except ValueError as ve:
            print(f"세션 ID 처리 오류: {str(ve)}")
            return await generate_chat_response(session_id, user_input, default_prompt)
            
    except Exception as e:
        print(f"챗봇 처리 중 오류 발생: {str(e)}")
        return {
            "content": "죄송합니다. 일시적인 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
            "authorType": "AI"
        }



async def generate_chat_response(session_id: int, user_input: str, system_prompt: str):
    """실제 챗봇 응답을 생성하는 함수"""
    try:
        # 세션의 마지막 메시지와 이미지 URL 조회
        latest_message_data = get_latest_message(int(session_id))
        if not latest_message_data["content"]:
            return {
                "content": "죄송합니다. 해당 세션의 메시지를 찾을 수 없습니다.",
                "authorType": "AI"
            }

        # 세션 메모리 가져오기
        _, memory = get_session_memory(session_id)

        # 이미지가 있는 경우, 이미지 설명을 텍스트로 추가
        user_message = latest_message_data["content"]
        if latest_message_data["image_url"]:
            image_description = await get_image_description(latest_message_data["image_url"])
            if image_description:
                user_message = f"{user_message}\n[이미지 설명: {image_description}]"

        # 메모리에서 대화 기록과 요약 로드
        memory_variables = memory.load_memory_variables({})
        chat_history = memory_variables.get("chat_history", "")
        conversation_summary = memory_variables.get("conversation_summary", "")

        # 텍스트 기반 응답 생성을 위한 프롬프트 템플릿 수정
        custom_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("system", "이전 대화 요약: {conversation_summary}"),
            ("system", "최근 대화 기록: {chat_history}"),
            ("user", "{user_message}")
        ])

        input_message = custom_prompt.format_messages(
            user_message=user_message,
            chat_history=chat_history,
            conversation_summary=conversation_summary
        )
        response = chat_model.invoke(input_message)

        # 응답이 비어있는 경우 처리
        if not response.content or not response.content.strip():
            return {
                "content": "죄송합니다. 다시 한 번 말씀해 주시겠어요?",
                "authorType": "AI"
            }

        # 메모리 업데이트
        memory.save_context({"input": user_message}, {"output": response.content})
        
        return {
            "content": clean_response(response.content),
            "authorType": "AI"
        }

    except Exception as e:
        print(f"응답 생성 중 오류 발생: {str(e)}")
        return {
            "content": "죄송합니다. 응답 생성 중 오류가 발생했습니다.",
            "authorType": "AI"
        }
        
    

# 통합
async def process_rag_ors_chat(session_id, user_input):
    try:
        print(f"[Step 1] session_id {session_id}에 대한 메시지 가져오는 중...")
        documents = fetch_user_messages_by_session(session_id)
        if not documents:
            return "해당 session_id로 가져올 메시지가 없습니다."

        print(f"[Step 2] 가져온 메시지로 ChromaDB 초기화 중...")
        initialize_chroma_with_supabase_data(session_id)

        print(f"[Step 3] ORS 점수 기반으로 적절한 프롬프트 선정 중...")
        # chat_with_bot이 이미 최종 응답을 반환하므로, 바로 반환하면 됩니다
        bot_response = await chat_with_bot(session_id, user_input)
        
        # 더 이상 문자열 체크나 추가 generate_chat_response 호출이 필요하지 않습니다
        return bot_response

    except Exception as e:
        print(f"프로세스 중 오류 발생: {e}")
        return {
            "content": "죄송합니다. 프로세스 중 오류가 발생했습니다. 관리자에게 문의하세요.",
            "authorType": "AI"
        }