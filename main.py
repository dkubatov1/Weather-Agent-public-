from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from os import getenv
import asyncio
import logging

from dotenv import load_dotenv
from openai import OpenAI
from langgraph.prebuilt import create_react_agent

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = ChatOpenAI(
  openai_api_key=getenv("OPENROUTER_API_KEY"),
  openai_api_base=getenv("OPENROUTER_BASE_URL"),
  model_name = "mistralai/mistral-7b-instruct"
)

checkpointer = InMemorySaver()

@tool
def get_weather(city: str) -> str:
    """ONLY returns 'The weather in {city} is sunny and 25°C.' Do not add any extra words."""
    return f"The weather in {city} is sunny and 25°C."

tools = [get_weather]

agent = create_react_agent(
    model=model,  
    tools=tools,
    prompt="You are a helpful assistant",
    checkpointer=checkpointer
)

async def main():
    thread_id = input("Enter a session ID (or press enter for default): ") or "default"
    config = {"configurable": {"thread_id": thread_id}}

    messages = []
    
    print("\nWelcome to the AI assistant. Type 'exit' to end the session.\n")

    while True:
        user_input = input("You: ").strip()
        
        if ('exit') in user_input.lower():
            print("Goodbye!")
            break

        messages = [{"role": "user", "content": user_input}]
        response = await agent.ainvoke({"messages": messages}, config)
        response["messages"][-1].pretty_print()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSession ended by user.")
