import os
import telebot
import warnings
import json
from io import BytesIO
from utils.assistant import VirtualAssistant

import utils.functions as functions
from utils.functions import *
from utils.utils import *

warnings.filterwarnings('ignore')

try:
    with open("utils/bot_info.json") as f_in:
        BOT_INFO = json.load(f_in)
    BOT_USERNAME = BOT_INFO["username"]
    BOT_TOKEN = BOT_INFO["token"]
except BaseException:
    print("You haven't provided Telegram bot info or your info is invalid")
    exit()
bot = telebot.TeleBot(BOT_TOKEN)
USER_SESSIONS = {}
model_id = "hiieu/Meta-Llama-3-8B-Instruct-function-calling-json-mode"
assistant = VirtualAssistant(llm_model_id=model_id,
                             llm_quantization=False,
                             memory_length=10)
assistant.system_prompt = SYSTEM_PROMPT
INIT_MESSAGE = {"role": "system", "content": assistant.system_prompt}
FUNCTIONS_TO_CONFIRM = {
    "process_absence_request": HRM_CHECKER_SYSTEM_PROMPT,
    "book_taxi": HRM_CHECKER_SYSTEM_PROMPT
    }
# ----------------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------------------------------------- #


def verify_user_request(messages, function_name):
    """
    Verify and confirm information before calling function
    """
    checker_assistant = VirtualAssistant(llm_model_id=model_id,
                                         llm_quantization=True)
    check_messages = messages
    check_messages[0] = {
        "role": "system",
        "content": FUNCTIONS_TO_CONFIRM[function_name]
    }
    check_result = checker_assistant.complete(check_messages)
    checker_assistant.release_gpu_memory()
    del checker_assistant
    return check_result


@bot.message_handler(func=lambda message: message.chat.id not in USER_SESSIONS)
def init_session(message):
    global USER_SESSIONS
    USER_SESSIONS[message.chat.id] = {
        "active": True,
        "dialogue": [INIT_MESSAGE]
    }
    if message.chat.type == 'private':
        bot.send_message(
            message.chat.id,
            f"Hi {message.from_user.first_name} 🤗, {BOT_INFO['name']} is here!")


@bot.message_handler(content_types=['sticker', 'audio'])
def refuse_reply(message):
    global USER_SESSIONS
    if message.chat.type == 'private':
        bot.reply_to(message, "I can't get it 🥺")
        pass


@bot.message_handler(content_types=['photo'])
def process_image(message):
    global USER_SESSIONS
    chat_id = message.chat.id
    user_stuffs_path = f"stuffs/user_{chat_id}"
    if not os.path.exists(user_stuffs_path):
        os.makedirs(user_stuffs_path)
    photo = message.photo[-1]
    file_info = bot.get_file(photo.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    file_path = os.path.join(user_stuffs_path, 'temp.jpg')
    with open(file_path, 'wb') as new_file:
        new_file.write(downloaded_file)
    USER_SESSIONS[chat_id]["dialogue"].append({
        "role": "user", "content": f"Path to the image is '{file_path}'"
    })


@bot.message_handler(commands=['bye'])
def terminate_session(message):
    global USER_SESSIONS
    chat_id = message.chat.id
    if chat_id in USER_SESSIONS:
        bot.send_message(chat_id, "Goodbye 🥺")
        USER_SESSIONS[message.chat.id] = {
            "active": False,
            "dialogue": []
        }
        USER_SESSIONS.pop(message.chat.id, None)
    else:
        pass


@bot.message_handler(func=lambda message:
                     USER_SESSIONS[message.chat.id]["active"])
def handle_active_bot(message):
    global USER_SESSIONS
    chat_id = message.chat.id
    if len(USER_SESSIONS[chat_id]["dialogue"]) > assistant.memory_length:
        USER_SESSIONS[chat_id]["dialogue"] = USER_SESSIONS[chat_id]["dialogue"][-assistant.memory_length:]
        USER_SESSIONS[chat_id]["dialogue"][0] = INIT_MESSAGE

    if message.chat.type == 'private':
        input_text = message.text
        USER_SESSIONS[chat_id]["dialogue"].append({"role": "user", "content": input_text})
        answer = assistant.complete(USER_SESSIONS[chat_id]["dialogue"])
        if "<functioncall>" in answer:
            function_info = get_function_info(answer)
            if function_info["name"] in FUNCTIONS_TO_CONFIRM.keys():
                check_result = verify_user_request(USER_SESSIONS[chat_id]["dialogue"],
                                                   function_info["name"])
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
        USER_SESSIONS[chat_id]["dialogue"].append({"role": "assistant", "content": answer})


if __name__ == '__main__':
    print("Bot is running")
    bot.infinity_polling()
