# main.py
# import uvicorn
from typing import List, Tuple

# import gradio as gr
from fastapi import FastAPI
from huggingface_hub import login

from agent.llm_agent import LlmAgent
from config.config import settings

# from app.api.routes import router

# login() # You will be prompted for your HF key, which will then be saved locally


agent = LlmAgent()
# agent.invoke("What is the weather in Stockholm today?")
agent.stream_agent("What is the weather in Stockholm today?")


def respond(message: str, history: List[Tuple[str, str]]) -> str:
    # Convert history to a format your agent can use if needed
    response = agent.invoke(message)
    # response = agent.stream_agent(message)
    return response


# Create the interface for gradio
# demo = gr.ChatInterface(
#     fn=respond,
#     title="Jens LLM Agent",
#     description="Jens AI assistant",
#     examples=["What's the current time?", "Tell me about the weather in Stockholm"],
# )


# def create_fastapi_app() -> FastAPI:
#     """Create the app to run."""
#     print(f"Model used is {settings.ollama_model}")
#     app = FastAPI(title="LLM Agent API")

#     # Initialize LLM Agent
#     # agent = LlmAgent()

#     agent.stream_agent("What is the weather in Stockholm today?")

#     # Add routes
#     # app.include_router(router)

#     return app


# app = create_fastapi_app()

# if __name__ == "__main__":
#     demo.launch(share=False, server_name="0.0.0.0")
    # uvicorn.run(app, host="0.0.0.0", port=8000)
