# Import relevant functionality
# import torch
import logging
import os
from uuid import uuid4

from config.config import settings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
# from langchain_huggingface import ChatHuggingFace
# from langchain_huggingface.llms import HuggingFacePipeline
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore

from agent.llm import Llm

# from transformers import (AutoModelForCausalLM, AutoTokenizer,
#                           BitsAndBytesConfig, pipeline)


# TODO Remove later
os.environ["TAVILY_API_KEY"] = settings.tavily_api_key
logger = logging.getLogger(__name__)

class LlmAgent:
    """
    LLMAgent class.

    agent docs:
    https://python.langchain.com/docs/tutorials/agents/#create-the-agent
    """

    def __init__(self):
        """Init function for LLMAgent."""
        # Memory api reference https://langchain-ai.github.io/langgraph/reference/checkpoints/#langgraph.checkpoint.memory.MemorySaver
        self.memory = MemorySaver()
        memory = InMemoryStore()
        # Tool api reference https://api.python.langchain.com/en/latest/tools/langchain_core.tools.tool.html
        # search_tool = TavilySearchResults(max_results=2)
        # self.tools = [search_tool]
        self.tools = []
        self.llm = Llm()
        self.chat_model = self.llm.chat_model
        # Creates a graph that works with a chat model that utilizes tool calling.
        self.agent_executor = create_react_agent(self.chat_model,
                                                 self.tools,
                                                 checkpointer=self.memory)
        self.config = {"configurable": {"thread_id": f"thread_{str(uuid4())}"}}

    def invoke(self, msg):
        """Invoke the llm agent."""
        response = self.agent_executor.invoke(
            {"messages": msg},
            self.config
        )
        print(response["messages"])
        logger.info(f"Response from agent: {response}")
        return response

    def invoke_llm(self):
        """Invoke the llm model."""
        res = self.chat_model.invoke([HumanMessage(content="hi!")])
        print(res.content)

    def start_new_conversation(self):
        """Invoke a new conversation with llm agent."""
        self.config["configurable"]["thread_id"] = f"thread_{str(uuid4())}"

    def stream_agent(self, humanMessage: str):
        """Stream llm agent."""
        for chunk in self.agent_executor.stream(
                {"messages": [HumanMessage(content=humanMessage)]},
                self.config
        ):
            print(chunk)
            print("----")


@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    return first_int * second_int


@tool
def add(first_int: int, second_int: int) -> int:
    """Add two integers."""
    return first_int + second_int


@tool
def exponentiate(base: int, exponent: int) -> int:
    """Exponentiate the base to the exponent power."""
    return base**exponent
