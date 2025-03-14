# main.py
# import uvicorn
import logging
from typing import List, Tuple

import gradio as gr
from fastapi import FastAPI
from huggingface_hub import login
from langchain_core.messages import AIMessage, HumanMessage

from agent.llm_agent import LlmAgent
from config.config import settings

# from app.api.routes import router

# login() # You will be prompted for your HF key, which will then be saved locally

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

agent = LlmAgent()
# agent.invoke("What is the weather in Stockholm today?")
# agent.stream_agent("What is the weather in Stockholm today?")


def respond(message: str, history: List[Tuple[str, str]]) -> str:
    history_langchain_format = []
    for msg in history:
        if msg['role'] == "user":
            history_langchain_format.append(
                HumanMessage(content=msg['content'])
            )
        elif msg['role'] == "assistant":
            history_langchain_format.append(
                AIMessage(content=msg['content'])
            )
    history_langchain_format.append(HumanMessage(content=message))

    response = agent.invoke(history_langchain_format)

    # return response.content
    # return response["messages"]
    return response["messages"][-1].content if response["messages"] \
        else "No response"

# Define a Gruvbox dark theme by extending Gradio's base theme
# gruvbox_theme = gr.themes.Base(
#     # Primary colors (buttons, links, etc.)
#     primary_hue=gr.themes.colors.blue,

#     # Secondary colors (highlights, accents)
#     secondary_hue=gr.themes.colors.orange,

#     # Neutral colors (background, text)
#     neutral_hue=gr.themes.colors.gray,

#     # Make the theme dark
#     font=[gr.themes.GoogleFont("Roboto"), "ui-sans-serif", "sans-serif"],
#     font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
# ).set(
#     # Customize specific elements to match Gruvbox colors
#     body_background_fill="#282828",
#     background_fill_primary="#3c3836",
#     background_fill_secondary="#504945",
#     border_color_primary="#665c54",
#     block_title_text_color="#ebdbb2",
#     block_label_text_color="#d5c4a1",
#     input_background_fill="#3c3836",
#     button_primary_background_fill="#458588",
#     button_primary_background_fill_hover="#83a598",
#     button_primary_text_color="#fbf1c7",
#     button_secondary_background_fill="#d65d0e",
#     button_secondary_background_fill_hover="#fe8019",
#     button_secondary_text_color="#fbf1c7",
#     # text_color="#ebdbb2",
# )

# # A minimal amount of custom CSS just for chat-specific elements
# chat_css = """
# .bot-message {
#     background-color: #3c3836 !important;
#     border: 1px solid #504945 !important;
# }
# .user-message {
#     background-color: #504945 !important;
#     border: 1px solid #665c54 !important;
# }
# """

# Create the interface for gradio
demo = gr.ChatInterface(
    fn=respond,
    type="messages",
    title="Jens LLM Agent",
    description="Jens AI assistant",
    examples=["What's the current time?", "Tell me about the weather in Stockholm"],
    # css=chat_css,         # Minimal CSS just for chat bubbles
    # theme=gruvbox_theme,  # Apply the Gruvbox theme
)


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

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0")
    # uvicorn.run(app, host="0.0.0.0", port=8000)
