import json
from utils.assistant import VirtualAssistant, OLlama_Assistant

from utils.functions import *
from utils.utils import *

try:
    with open("utils/bot_info.json") as f_in:
        BOT_INFO = json.load(f_in)
    BOT_USERNAME = BOT_INFO["username"]
    BOT_TOKEN = BOT_INFO["token"]
except BaseException:
    print("You haven't provided Telegram bot info")
    exit()
FUNCTIONS_TO_CONFIRM = {
    "process_absence_request": HRM_CHECKER_SYSTEM_PROMPT,
    "book_taxi": HRM_CHECKER_SYSTEM_PROMPT
}

master_bot_info = BOT_INFO["assistants"]["master"]
# assistant = VirtualAssistant(llm_model_id=master_bot_info["model_id"],
#                              llm_quantization=False,
#                              llm_use_bitsandbytes=False,
#                              llm_max_tokens=1024,
#                              memory_length=10)
# assistant.system_prompt = SYSTEM_PROMPT

assistant = OLlama_Assistant()
assistant.system_prompt = SYSTEM_PROMPT

# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# checker = VirtualAssistant(llm_model_id=model_id,
#                            llm_quantization=True,
#                            llm_use_bitsandbytes=True,
#                            llm_max_tokens=128)

INIT_MESSAGE = {"role": "system", "content": assistant.system_prompt}
