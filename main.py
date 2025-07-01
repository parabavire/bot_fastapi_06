from fastapi import FastAPI
from pydantic import BaseModel
from langchain_cohere import ChatCohere
from langchain_core.messages import SystemMessage, HumanMessage

from dotenv import load_dotenv
load_dotenv()

app = FastAPI()
model = ChatCohere(model="command-r-plus")

system_prompt_filepath = "system_message-06.txt"
with open(system_prompt_filepath, "r", encoding="utf-8") as f:
    system_prompt_content = f.read()

messages = [
    SystemMessage(content=system_prompt_content)
]

@app.get("/")
async def root():
    return {"message": "Hello World"}

class Bot(BaseModel):
    query: str

@app.post("/bot")
async def bot(q: Bot):

    question = q.query
    print(f"Usuario: {question}")
    messages.append(HumanMessage(content=question)) 
    
    print(f"Generando respuesta")
    response = model.invoke(messages) 
    
    messages.append(response)
    print(f"Bot: {response.content}")
    return response.content
