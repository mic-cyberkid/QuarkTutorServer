# chat_llm_service.py
import os
import logging # Keep this import for type hinting or if you need default logging somewhere else
from chat_utils import SYSTEM_PROMPT
from llama_cpp import Llama
from tech_support_logger import TechSupportLogger # Import your custom logger

# --- Initialize the custom logger for this module ---
# For simplicity and to ensure logging works even if this module is run independently,
# we'll initialize a logger here. The TechSupportLogger class handles
# ensuring only one set of handlers is configured for the root logger.
# We'll set it to INFO here, but the main FullServer.py will control the overall level.
llm_logger = TechSupportLogger(
    log_file_name="llm_service.log", # Separate log for LLM specific issues
    log_dir="data/logs",
    level=logging.INFO,
    max_bytes=10 * 1024 * 1024, # 10 MB
    backup_count=5,
    console_output=False # Typically, LLM service doesn't need to print to console
).get_logger()


MODEL_DIR = "models/"
MODEL_FILENAME = "PhysicsChatBot.gguf" # GGUF File
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)




MODEL_CONFIGS = {
    "qwen1_5-0.5B-Q8_0.gguf": "chatml",
    "tinyllama-1.1b-q6.gguf": "llama-2",
    "mistral-7b-q4.gguf": "mistral-instruct",
    # Add more...
}

GENERATION_PARAMS = {
    "stop": ["<|im_end|>"],
    "max_tokens": 2048,
    "temperature": 0.7,
    "top_p": 0.9,
    "repeat_penalty": 1.1,

}


class ChatLLMService:
    def __init__(self, model_path: str = MODEL_PATH):
        self.model = Llama(
            model_path=model_path,
            #chat_format="chatml",
            n_ctx=20048,
            n_gpu_layers=0,
            use_mmap=True,
            verbose=False,
        )
        # Use the new logger
        llm_logger.info("LLM model loaded from: %s", model_path)

    def format_prompt(self, message, history, curriculum_chunks):
        prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"

        prompt += f"<|im_start|>user\nContext:\n" + "\n---\n".join(curriculum_chunks[:2]) + "<|im_end|>\n"

        # Use only last 2 turns
        for msg in history[-2:]:
            prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"

        prompt += f"<|im_start|>user\n{message}\n\n<|im_end|>\n<|im_start|>assistant:\n<think>\n\n</think>\n\n"

        return prompt


    def generate_response(self, prompt: str) -> str:
        try:
            output = self.model(prompt, **GENERATION_PARAMS)
            return output['choices'][0]['text'].strip()
        except Exception as e:
            # Use the new logger, adding exc_info=True to capture traceback
            llm_logger.error("LLM generation error: %s", e, exc_info=True)
            raise RuntimeError("Failed to generate LLM response.") from e

    def generate_response_stream(self, prompt: str):
        """
        Generator that streams LLM output token-by-token.
        Yields plain text tokens (or wrap with `data: ...\n\n` for SSE).
        """
        def token_stream():
            try:
                for output in self.model(prompt, stream=True, **GENERATION_PARAMS):
                    token = output["choices"][0]["text"]
                    yield token  # Or yield f"data: {token}\n\n" for SSE
            except Exception as e:
                # Use the new logger, adding exc_info=True to capture traceback
                llm_logger.error("LLM streaming error: %s", e, exc_info=True)
                yield "[ERROR] Streaming failed.\n"

        return token_stream()
