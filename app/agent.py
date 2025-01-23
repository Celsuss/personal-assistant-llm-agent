# Import relevant functionality
import torch
from huggingface_hub import login
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_huggingface.llms import HuggingFacePipeline
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, pipeline)

# import llm

# login() # You will be prompted for your HF key, which will then be saved locally

# Configure model loading with 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Initialize tokenizer and model
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
)

# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10)
pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=2048,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.15,
    return_full_text=True,
)
hf = HuggingFacePipeline(pipeline=pipeline)

chat_model = ChatHuggingFace(llm=hf, verbose=True)


# messages = [
#     ("system", "You are a helpful translator. Translate the user sentence to Swedish."),
#     ("human", "I love programming."),
# ]

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
