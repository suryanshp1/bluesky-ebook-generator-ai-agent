from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

system = "You are a helpful assistant."
human = "{text}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

chat = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY1"), model_name="mixtral-8x7b-32768", max_tokens=250)

chain = prompt | chat
x = chain.invoke({"text": "Explain the importance of low latency LLMs."})

print(x.content)