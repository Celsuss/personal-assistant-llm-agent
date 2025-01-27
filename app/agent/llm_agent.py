# Import relevant functionality
# import torch
# from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
# from langchain_huggingface import ChatHuggingFace
# from langchain_huggingface.llms import HuggingFacePipeline
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from agent.llm import Llm

# from transformers import (AutoModelForCausalLM, AutoTokenizer,
#                           BitsAndBytesConfig, pipeline)


class LlmAgent:
    """
    LLMAgent class.

    agent docs:
    https://python.langchain.com/docs/tutorials/agents/#create-the-agent
    """

    def __init__(self, settings):
        """Init function for LLMAgent."""
        self.settings = settings
        # Memory api reference https://langchain-ai.github.io/langgraph/reference/checkpoints/#langgraph.checkpoint.memory.MemorySaver
        self.memory = MemorySaver()
        # Tool api reference https://api.python.langchain.com/en/latest/tools/langchain_core.tools.tool.html
        self.tools = [multiply, add, exponentiate]
        self.llm = Llm()
        self.chat_model = self.llm.chat_model
        self.agent_executor = create_react_agent(self.chat_model,
                                                 self.tools,
                                                 checkpointer=self.memory)
        self.config = {"configurable": {"thread_id": "abc123"}}

    def invoke(self):
        """Invoke the llm agent."""
        response = self.agent_executor.invoke(
            {"messages": [HumanMessage(content="hi!")]},
            self.config
        )
        print(response["messages"])

    def invoke_llm(self):
        """Invoke the llm model."""
        res = self.chat_model.invoke([HumanMessage(content="hi!")])
        print(res.content)

    def start_conversation(self):
        """Invoke a new conversation with llm agent."""
        self.config["configurable"]["thread_id"] + "bca321" # Add a random function here

    def stream_agent(self):
        """Stream llm agent."""
        for chunk in self.agent_executor.stream(
                {"messages": [HumanMessage(content="hi im bob! and i live in sf")]},
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
