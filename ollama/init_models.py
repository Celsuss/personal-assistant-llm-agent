import logging
import sys
import time
from typing import List

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaManager:
    def __init__(self, host: str = "http://ollama:11434"):
        self.host = host
        # TODO Load from settings.toml
        self.models = ["mistral:7b", "deepseek-r1:7b"]  # Add your models here

    def wait_for_ollama(self, timeout: int = 60) -> bool:
        """Wait for Ollama service to become available."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.host}/api/tags")
                if response.status_code == 200:
                    logger.info("Ollama service is available")
                    return True
            except requests.exceptions.RequestException:
                logger.info("Waiting for Ollama service...")
                time.sleep(2)

        logger.error(f"Ollama service not available after {timeout} seconds")
        return False

    def pull_model(self, model_name: str) -> bool:
        """Pull a specific model."""
        try:
            logger.info(f"Pulling model: {model_name}")
            response = requests.post(
                f"{self.host}/api/pull",
                json={"name": model_name}
            )

            if response.status_code == 200:
                logger.info(f"Successfully pulled {model_name}")
                return True
            else:
                logger.error(f"Failed to pull {model_name}: {response.text}")
                return False

        except requests.exceptions.RequestException as e:
            logger.error(f"Error pulling {model_name}: {str(e)}")
            return False

    def pull_all_models(self) -> None:
        """Pull all specified models."""
        if not self.wait_for_ollama():
            sys.exit(1)

        for model in self.models:
            if not self.pull_model(model):
                logger.error(f"Failed to pull {model}")
                sys.exit(1)


if __name__ == "__main__":
    manager = OllamaManager()
    manager.pull_all_models()
