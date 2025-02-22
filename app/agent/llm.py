import os

import torch
from config.config import settings
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface.llms import HuggingFaceEndpoint, HuggingFacePipeline
from langchain_ollama import ChatOllama
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, pipeline)


class Llm:
    """Huggingface LLM model class."""

    def __init__(self):
        """Init function for LLM class."""
        # self.loadHuggingFacePipeline()
        self.loadOllama()

    def loadOllama(self):
        """Load Ollama model.

        Api docs:
        https://python.langchain.com/api_reference/ollama/chat_models/langchain_ollama.chat_models.ChatOllama.html#langchain_ollama.chat_models.ChatOllama
        """
        self.chat_model = ChatOllama(
            model=settings.ollama_model,
            temperature=0.6,
            num_predict=256,
            base_url=os.environ["OLLAMA_HOST", 'http://localhost:11434']
            # other params ...
        )


    def loadHuggingFacePipeline(self):
        """Load huggingFacePipeline chat model.

        Api docs:
        https://python.langchain.com/api_reference/huggingface/chat_models/langchain_huggingface.chat_models.huggingface.ChatHuggingFace.html#langchain_huggingface.chat_models.huggingface.ChatHuggingFace
        """
        # Configure model loading with 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        # Initialize tokenizer and model
        model_name = settings.hf_model_path
        # model_name = "mistralai/Mistral-7B-Instruct-v0.1"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=2048,
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.15,
            return_full_text=True,
            truncation=True,
            do_sample=True,
            # max_new_tokens=10,
        )
        hf = HuggingFacePipeline(pipeline=pipe)

        self.chat_model = ChatHuggingFace(llm=hf, verbose=True)
