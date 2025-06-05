from transformers import pipeline, AutoTokenizer
import tiktoken
import torch
from groq import Groq

class Summarizer_HuggingFace:
    def __init__(self, model_name: str):
        device = 0 if torch.cuda.is_available() else -1
        self.pipeline = pipeline("summarization", model=model_name, tokenizer=model_name, device=device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_input_tokens = 1024

    def _chunk_texts(self, text: str):
        tokens = self.tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
        chunks = [tokens[i:i+self.max_input_tokens] for i in range(0, len(tokens), self.max_input_tokens)]
        return [self.tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

    def __call__(self, chunks: list, max_length: int, min_length: int, do_sample: bool) -> str:
        summaries = []
        for chunk in chunks:
            try:        
                summary = self.pipeline(
                    chunk,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=do_sample,
                    truncation=True
                )[0]['summary_text']
                summaries.append(summary)
            except Exception as e:
                print(f"Error summarizing chunk: {e}")
        return "\n".join(summaries)
    
class Summarizer_Groq:
    def __init__(self, model_name: str, api_key: str):
        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        self.max_input_tokens = 4000
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        except Exception:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str):
        return len(self.tokenizer.encode(text))

    def __call__(self, text: str, max_tokens: int, temperature: float, top_p: float) -> str:
        if self._count_tokens(text) > self.max_input_tokens:
            text = self.tokenizer.decode(self.tokenizer.encode(text)[:self.max_input_tokens])
        messages = [{"role": "user", "content": text}]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        ).choices[0].message.content.strip()
        return response