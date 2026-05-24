import torch
from diffusers import AutoPipelineForText2Image
from prompt_engine import rewriter
import gc


class DreamFrameEngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DreamFrameEngine, cls).__new__(cls)
            cls._instance.device = "cuda" if torch.cuda.is_available() else "cpu"
            cls._instance.pipe = None
            cls._instance.current_model_id = None
        return cls._instance

    def load_model(self, model_name):
        models = {
            "Fast Mode": "stabilityai/sd-turbo",
            "Artistic Mode": "Lykon/dreamshaper-xl-1-0",
            "Realistic Mode": "SG161222/RealVisXL_V4.0"
        }

        # Default fallback
        model_id = models.get(model_name, "Lykon/dreamshaper-xl-1-0")

        if self.current_model_id != model_id:
            self.pipe = None
            gc.collect()
            if self.device == "cuda": torch.cuda.empty_cache()

            self.pipe = AutoPipelineForText2Image.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_safetensors=True
            ).to(self.device)
            self.current_model_id = model_id

    def generate(self, prompt, steps, scale, w, h, model_name):
        # 1. Prompt ko rewrite karein (Auto-Expand)
        expanded_prompt = rewriter.rewrite_prompt(prompt)

        # 2. Model load karein
        self.load_model(model_name)

        # Negative Prompt (User ko dikhane ki zarorat nahi)
        negative_prompt = "blurry, low quality, distorted, bad anatomy, ugly, extra limbs, bad hands, text, cropped"

        # Performance & memory optimization combinations
        if torch.cuda.is_available():
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.device)
            self.pipe.enable_attention_slicing()

        return self.pipe(
            prompt=expanded_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=int(steps),
            guidance_scale=scale,
            width=w, height=h
        ).images[0]