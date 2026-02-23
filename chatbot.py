from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
     repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="task-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm = llm)

chat_messages = [
    SystemMessage(content="You are helpful AI assistant.")
]

while True:
    user_input = input("You: ")
    chat_messages.append(HumanMessage(content=user_input))
    if user_input=="exit":
        break
    result = model.invoke(chat_messages)
    chat_messages.append(AIMessage(content=result.content))
    print(f"AI: {result.content}")