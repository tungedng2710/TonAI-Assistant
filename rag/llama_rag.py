import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

ST = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

dataset = load_dataset("not-lain/wikipedia")

def embed(batch):
    """
    adds a column to the dataset called 'embeddings'
    """
    # or you can combine multiple columns here
    # For example the title and the text
    information = batch["text"]
    return {"embeddings": ST.encode(information)}


dataset = dataset.map(embed, batched=True, batch_size=256)
data = dataset["train"]
data = data.add_faiss_index("embeddings")


def search(query: str, k: int = 3):
    """a function that embeds a new query and returns the most probable results"""
    embedded_query = ST.encode(query)  # embed new query
    scores, retrieved_examples = data.get_nearest_examples(  # retrieve results
        # compare our new embedded query with the dataset embeddings
        "embeddings", embedded_query,
        k=k  # get only top k results
    )
    return scores, retrieved_examples


model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_id = "bkai-foundation-models/vietnamese-llama2-7b-40GB"
# use quantization to lower GPU usage
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # torch_dtype=torch.bfloat16,
    device_map="auto",
    # quantization_config=bnb_config
)
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

SYS_PROMPT = """You are an assistant for answering questions.
You are given the extracted parts of a long document and a question. Provide a conversational answer.
If you don't know the answer, just say "I do not know." Don't make up an answer."""


def format_prompt(prompt, retrieved_documents, k):
    """using the retrieved documents we will prompt the model to generate our responses"""
    PROMPT = f"Question:{prompt}\nContext:"
    for idx in range(k):
        PROMPT += f"{retrieved_documents['text'][idx]}\n"
    print(f"prompt length: {len(PROMPT)}")
    return PROMPT


def generate(prompt: str = ""):
    prompt = prompt[:10000]  # to avoid GPU OOM
    messages = [{"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": prompt}]
    # tell the model to generate
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    outputs = model.generate(
        input_ids,
        max_new_tokens=1024,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)


def rag_chatbot(prompt: str, k: int = 2, use_rag: bool = True):
    scores, retrieved_documents = search(prompt, k)
    if use_rag:
        formatted_prompt = format_prompt(prompt, retrieved_documents, k)
    else:
        formatted_prompt = prompt
    return generate(formatted_prompt)


while True:
    prompt = input("Prompt: ")
    response = rag_chatbot(prompt, k=4, use_rag=True)
    print(f"Response: {response}")