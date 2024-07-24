import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class RAG:
    def __init__(self) -> None:
        self.bnb_config = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.bfloat16)
        self.init_llm()
        self.instruction = """You are an assistant for answering questions.
                            You are given the extracted parts of a long document and a question. Provide a conversational answer.
                            If you don't know the answer, just say "I do not know." Don't make up an answer.
                            """
        

    def init_llm(self, 
                 model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct", 
                 quantization: bool = False):
        """
        Init LLM model
        """
        if quantization:
            self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    quantization_config=self.bnb_config
                )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)


