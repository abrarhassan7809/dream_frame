import torch
from transformers import pipeline


class PromptRewriter:
    def __init__(self):
        # TinyLlama ek fast aur powerful model hai prompt rewriting ke liye
        self.pipe = pipeline(
            "text-generation",
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

    def rewrite_prompt(self, user_prompt):
        system_msg = (
            "You are an expert AI art prompt engineer. Rewrite the following user prompt "
            "into a detailed, cinematic prompt for Stable Diffusion. Focus on lighting, "
            "composition, and anatomical correctness. Only return the prompt, no explanations."
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Rewrite this: {user_prompt}"},
        ]

        # Prompt format for TinyLlama
        prompt = self.pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        outputs = self.pipe(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)
        rewritten = outputs[0]["generated_text"].split("<|assistant|>\n")[-1]

        return rewritten.strip()


# Initialize rewriter
rewriter = PromptRewriter()