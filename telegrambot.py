import os
import torch
import telebot
import warnings
from io import BytesIO
from g4f.client import Client
from assistant import VirtualAssistant

import functions
from functions import *
from utils.utils import get_function_info

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
messages = [
    {"role": "system", "content": assistant.system_prompt},
]


@bot.message_handler(func=lambda message: message.chat.id not in USER_SESSIONS)
def add_new_user(message):
    global USER_SESSIONS
    # Initialize user session
    USER_SESSIONS[message.chat.id] = {"active": True}
    USER_SESSIONS[message.chat.id]["dialogue"] = [
        {"role": "system", "content": assistant.system_prompt},
    ]
    user_name = message.from_user.first_name
    if message.chat.type == 'private':
        bot.send_message(
            message.chat.id,
            f"Hi {user_name} 🤗, How can I help you today")
    else:
        pass

@bot.message_handler(content_types=['sticker', 'audio'])
def refuse_reply(message):
    global USER_SESSIONS
    user_session = USER_SESSIONS[message.chat.id]
    if user_session["active"]:
        if message.chat.type == 'private':
            bot.reply_to(message, "I cannot process this content 🥺")
            pass
        else:
            pass


@bot.message_handler(func=lambda message:
                     USER_SESSIONS[message.chat.id]["active"])
def handle_active_bot(message):
    global USER_SESSIONS
    user_session = USER_SESSIONS[message.chat.id]

    if user_session['active']:
        if len(user_session["dialogue"]) > assistant.memory_length:
            user_session["dialogue"] = user_session["dialogue"][-assistant.memory_length:]
            user_session["dialogue"][0] = {
                "role": "system", "content": assistant.system_prompt}

        if message.chat.type == 'private':
            chat_id = message.chat.id
            input_text = message.text
            user_session["dialogue"].append(
                {"role": "user", "content": input_text})
            answer = assistant.complete(user_session["dialogue"])
            if "<functioncall>" in answer:
                function_info = get_function_info(answer)
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
                bot.send_message(chat_id, answer)
            user_session["dialogue"].append(
                {"role": "assistant", "content": answer})
        # else:
        #     if f"@{BOT_USERNAME}" in message.text:
        #         input_text = message.text.replace(f"@{BOT_USERNAME}", "")
        #         user_session["dialogue"].append(
        #             {"role": "user", "content": input_text})
        #         answer = assistant.complete(user_session["dialogue"])
        #         user_session["dialogue"].append(
        #             {"role": "assistant", "content": answer})
        #         bot.reply_to(message, answer)
        #     else:
        #         pass


if __name__ == '__main__':
    print("Bot is running")
    bot.infinity_polling()
