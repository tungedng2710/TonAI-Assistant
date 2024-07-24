import re
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import functions
from functions import *

# Suppress warnings
warnings.filterwarnings("ignore")


def get_function_info(answer: str = ""):
    match = re.search(r'<functioncall>(.*?)</functioncall>', answer)
    if match:
        functioncall_string = match.group(1).strip()
        functioncall_string = functioncall_string.replace("'", '')
        dictionary = json.loads(functioncall_string)
        return dictionary
    else:
        return


def complete(model, tokenizer, messages, max_new_tokens=1024):
    """
    Generate text with LLMs
    """
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=terminators,
        pad_token_id=128001,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)


if __name__ == "__main__":
    model_id = "hiieu/Meta-Llama-3-8B-Instruct-function-calling-json-mode"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16
    )
    functions_metadata = FUNCTIONS_METADATA

    system_prompt = f"""You are the virtual assistant of TonAI with access to the following functions:
    {str(functions_metadata)}\n\nTo use these functions respond with:
    <functioncall> {{ "name": "function_name", "arguments": {{ "arg_1": "value_1", "arg_1": "value_1", ... }} }} </functioncall>
    Edge cases you must handle:
    - If there are no functions that match the user request, you will respond politely that you cannot help.
    """

    messages = [
        {"role": "system", "content": system_prompt},
    ]
    while True:
        user_input = input("Prompt: ")
        messages.append({"role": "user", "content": user_input})
        answer = complete(model, tokenizer, messages)
        messages.append({"role": "assistant", "content": answer})
        if "<functioncall>" in answer:
            function_info = get_function_info(answer)
            function_to_call = getattr(functions, function_info["name"])
            function_to_call(**function_info["arguments"])
        else:
            print(f"TonAI Virtual Assistant: {answer}")
