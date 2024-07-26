import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Suppress warnings
warnings.filterwarnings("ignore")


class VirtualAssistant:
    def __init__(self,
                 memory_length: int = 20,
                 llm_model_id: str = None,
                 llm_quantization: bool = False,
                 llm_max_tokens: int = 512) -> None:
        self.system_prompt = ""
        self.memory_length = memory_length
        self.init_llm(llm_model_id, llm_quantization)
        self.llm_max_tokens = llm_max_tokens

    def init_llm(
            self,
            model_id: str = "",
            quantization: bool = False):
        """
        Initialize Large Language Model
        Args:
            - model_id (str) : path to HF model
            - quantization (bool) : use quantized model
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
                # "quantization_config": bnb_config,
                "torch_dtype": torch.bfloat16
            }
        else:
            quantization_args = {}
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            **quantization_args)

    def complete(self, messages):
        """
        Generate text with LLMs
        Args:
            - messages: list of messages with ChatML format
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
            max_new_tokens=self.llm_max_tokens,
            eos_token_id=terminators,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        del input_ids
        return self.tokenizer.decode(response, skip_special_tokens=True)
    
    def release_gpu_memory(self):
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()

    