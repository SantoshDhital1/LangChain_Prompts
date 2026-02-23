from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
     repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="task-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm = llm)

messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="tell me about langChain.")
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))
print(messages)