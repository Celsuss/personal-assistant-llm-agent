# main.py
# import uvicorn
import logging
from typing import Generator, List, Tuple

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


def respond(message: Tuple[str, str], history: List[Tuple[str, str]]) -> Generator[str, None, None]:
# def respond(message: Tuple[str, str], history: List[Tuple[str, str]]):
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
    history_langchain_format.append(HumanMessage(content=message['content']))


    # for chunk in agent.invoke_stream(history_langchain_format):
    #     yield chunk
    yield from agent.invoke_stream(history_langchain_format)


js_func = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""

with gr.Blocks(js=js_func) as demo:
    chatbot = gr.Chatbot(type="messages")  # Enable streaming
    msg = gr.Textbox()
    # clear = gr.Button("Clear")
    clear = gr.ClearButton([msg, chatbot])

    def user(user_message, history):
        return "", history + [{"role": "user", "content": user_message}]

    def bot(history):
        history.append({"role": "assistant", "content": ""})
        user_message = history[-1]

        current_content = ''
        for character in respond(user_message, history[:-1]):
            current_content += character['agent']['messages'][-1].text()
            history[-1] = {"role": "assistant", "content": current_content}
            yield history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

# demo.queue()
# demo.launch()

# Create the interface for gradio
# demo = gr.ChatInterface(
#     fn=respond,
#     type="messages",
#     title="Jens LLM Agent",
#     description="Jens AI assistant",
#     examples=["Say hi", "What's the current time?", "Tell me about the weather in Stockholm"],
#     js=js_func,
#     streaming=True,
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

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0")
    # uvicorn.run(app, host="0.0.0.0", port=8000)
