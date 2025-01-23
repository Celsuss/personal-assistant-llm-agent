# Import relevant functionality
import torch
from huggingface_hub import login
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface.llms import HuggingFacePipeline
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, pipeline)

import llm

# login() # You will be prompted for your HF key, which will then be saved locally
chat_model = llm.chat_model

# Create the agent
memory = MemorySaver()

@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    return first_int * second_int


@tool
def add(first_int: int, second_int: int) -> int:
    "Add two integers."
    return first_int + second_int


@tool
def exponentiate(base: int, exponent: int) -> int:
    "Exponentiate the base to the exponent power."
    return base**exponent


# Tool api reference https://api.python.langchain.com/en/latest/tools/langchain_core.tools.tool.html
tools = [multiply, add, exponentiate]

agent_executor = create_react_agent(chat_model, tools, checkpointer=memory)

# Use the agent
config = {"configurable": {"thread_id": "abc123"}}
for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="hi im bob! and i live in sf")]}, config
):
    print(chunk)
    print("----")
