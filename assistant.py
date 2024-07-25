import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import functions
from functions import *
from utils.utils import SYSTEM_PROMPT, get_function_info

# Suppress warnings
warnings.filterwarnings("ignore")


class VirtualAssistant:
    def __init__(self,
                 memory_length: int = 20,
                 llm_model_id: str = None,
                 llm_quantization: bool = False) -> None:
        self.system_prompt = SYSTEM_PROMPT
        self.memory_length = memory_length
        self.init_llm(llm_model_id, llm_quantization)

    def init_llm(
            self,
            model_id: str = "",
            quantization: bool = False):
        """
        Initialize Large Language Model
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            quantization_args = {
                "quantization_config": bnb_config,
                "torch_dtype": torch.bfloat16
            }
        else:
            quantization_args = {}
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            **quantization_args)

    def complete(self, messages, max_new_tokens=256):
        """
        Generate text with LLMs
        """
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(response, skip_special_tokens=True)


if __name__ == "__main__":
    model_id = "hiieu/Meta-Llama-3-8B-Instruct-function-calling-json-mode"
    assistant = VirtualAssistant(llm_model_id=model_id,
                                 llm_quantization=False,
                                 memory_length=10)
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
