# main.py
import uvicorn
from fastapi import FastAPI
from huggingface_hub import login

from agent.llm_agent import LlmAgent
from config.config import settings

# from app.api.routes import router

# login() # You will be prompted for your HF key, which will then be saved locally

def create_app() -> FastAPI:
    """Create the app to run."""
    print(f"Model used is {settings.ollama_model}")
    app = FastAPI(title="LLM Agent API")

    # Initialize LLM Agent
    agent = LlmAgent()

    agent.stream_agent("What is the weather in Stockholm today?")

    # Add routes
    # app.include_router(router)

    return app


app = create_app()

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
