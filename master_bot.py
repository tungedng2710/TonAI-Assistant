import os
import torch
import telebot
import warnings
from io import BytesIO
from g4f.client import Client
from utils.assistant import VirtualAssistant

import utils.functions as functions
from utils.functions import *
from utils.utils import get_function_info, SYSTEM_PROMPT

warnings.filterwarnings('ignore')

BOT_USERNAME = "tonai_chat_bot"

try:
   with open("utils/bot_token.txt") as txtfile:
      BOT_TOKEN = txtfile.read()
except:
   print("You haven't provided Telegram bot token!")
   print("Create your bot and add your token at utils/bot_token.txt")
   exit()
bot = telebot.TeleBot(BOT_TOKEN)
bot_active = True
client = Client()
USER_SESSIONS = {}
model_id = "hiieu/Meta-Llama-3-8B-Instruct-function-calling-json-mode"
assistant = VirtualAssistant(llm_model_id=model_id,
                             llm_quantization=False,
                             memory_length=10)
assistant.system_prompt = SYSTEM_PROMPT
INIT_MESSAGES = [
    {"role": "system", "content": assistant.system_prompt},
]


@bot.message_handler(func=lambda message: message.chat.id not in USER_SESSIONS)
def add_new_user(message):
    global USER_SESSIONS
    # Initialize user session
    USER_SESSIONS[message.chat.id] = {"active": True}
    USER_SESSIONS[message.chat.id]["dialogue"] = INIT_MESSAGES
    user_name = message.from_user.first_name
    if message.chat.type == 'private':
        bot.send_message(
            message.chat.id,
            f"Hi {user_name} 🤗, How can I help you today")


@bot.message_handler(content_types=['sticker', 'audio'])
def refuse_reply(message):
    global USER_SESSIONS
    if message.chat.type == 'private':
        bot.reply_to(message, "I cannot process this content 🥺")
        pass


@bot.message_handler(commands=['bye'])
def terminate_session(message):
    global USER_SESSIONS
    chat_id = message.chat.id
    if chat_id in USER_SESSIONS:
        bot.send_message(chat_id, "Session ended. Goodbye 🥺")
        del USER_SESSIONS[chat_id]
    else:
        pass


@bot.message_handler(func=lambda message:
                     USER_SESSIONS[message.chat.id]["active"])
def handle_active_bot(message):
    global USER_SESSIONS
    user_session = USER_SESSIONS[message.chat.id]

    if len(user_session["dialogue"]) > assistant.memory_length:
        user_session["dialogue"] = user_session["dialogue"][-assistant.memory_length:]
        user_session["dialogue"][0] = INIT_MESSAGES

    if message.chat.type == 'private':
        chat_id = message.chat.id
        input_text = message.text
        user_session["dialogue"].append({"role": "user", "content": input_text})
        answer = assistant.complete(user_session["dialogue"])
        if "<functioncall>" in answer:
            function_info = get_function_info(answer)
            if function_info["name"] == "process_absence_request":
                checker_assistant = VirtualAssistant(llm_model_id=model_id,
                                                     llm_quantization=True)
                check_messages = user_session["dialogue"]
                check_messages[0] = {"role": "system", "content": HRM_CHECKER_SYSTEM_PROMPT}
                check_result = checker_assistant.complete(check_messages)
                checker_assistant.release_gpu_memory()
                del checker_assistant
            else:
                check_result = "<accepted>"
            if "<accepted>" in check_result or "<functioncall>" in check_result:
                print("Request accepted, calling function...")
                function_to_call = getattr(functions, function_info["name"])
                result = function_to_call(**function_info["arguments"])
                datatype = result[-1]
                if datatype == "image":
                    image, image_url = result[0], result[1]
                    bio = BytesIO()
                    bio.name = image_url
                    image.save(bio, 'PNG')
                    bio.seek(0)
                    bot.send_photo(message.chat.id, photo=bio)
                    os.remove(image_url)
                else:
                    bot.send_message(chat_id, result[0])
            else:
                print("Request is not accepted")
                bot.send_message(chat_id, check_result)
        else:
            bot.send_message(chat_id, answer)
        user_session["dialogue"].append({"role": "assistant", "content": answer})


if __name__ == '__main__':
    print("Bot is running")
    bot.infinity_polling()
