import torch
from transformers import pipeline
from memory import PromptMemory

memory = PromptMemory()

class PromptRewriter:
    def __init__(self):
        # Corrected 'dtype' to 'torch_dtype'
        self.pipe = pipeline(
            "text-generation",
            model="microsoft/Phi-3-mini-4k-instruct",
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

    def rewrite_prompt(self, user_prompt):
        history_context = memory.analyze_history()
        system = f"""
You are an elite AI image prompt engineer.

Your tasks:
- Preserve ALL user details
- Understand scene relationships
- Improve realism, lighting, and cinematic composition
- Understand previous user preferences

Previous user preferences context:
{history_context}

Return ONLY the optimized visual prompt text description. Do not add conversational conversational filler.
"""

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt}
        ]

        prompt = self.pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        output = self.pipe(prompt, max_new_tokens=150, temperature=0.7, top_p=0.9, do_sample=True)

        result = output[0]["generated_text"]

        if "<|assistant|>" in result:
            result = result.split("<|assistant|>")[-1]

        # Clean prompt layout matching premium engines
        final_prompt = f"cinematic shot, highly detailed, sharp focus, {result.strip()}"

        memory.save_prompt(user_prompt)
        return final_prompt

rewriter = PromptRewriter()