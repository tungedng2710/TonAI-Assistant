import utils.functions as functions
from utils.assistant import VirtualAssistant
from utils.functions import *
from utils.utils import SYSTEM_PROMPT, get_function_info


if __name__ == "__main__":
    model_id = "hiieu/Meta-Llama-3-8B-Instruct-function-calling-json-mode"
    assistant = VirtualAssistant(llm_model_id=model_id,
                                 llm_quantization=False,
                                 memory_length=10)
    assistant.system_prompt = SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": assistant.system_prompt},
    ]
    while True:
        user_input = input("User: ")
        messages.append({"role": "user", "content": user_input})
        answer = assistant.complete(messages)
        messages.append({"role": "assistant", "content": answer})
        if len(messages) > assistant.memory_length:
            messages = messages[0] + messages[-assistant.memory_length:]
        if "<functioncall>" in answer:
            function_info = get_function_info(answer)
            function_to_call = getattr(functions, function_info["name"])
            function_to_call(**function_info["arguments"])
        else:
            print(f"TonAI Assistant: {answer}")
