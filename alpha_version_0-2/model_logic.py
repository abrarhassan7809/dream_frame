import os
import torch
from diffusers import FluxPipeline
from prompt_engine import rewriter
import gc

class DreamFrameEngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.device = "cuda" if torch.cuda.is_available() else "cpu"
            cls._instance.pipe = None
        return cls._instance

    def load_model(self):
        if self.pipe is not None:
            return

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        local_model = "./models/FLUX"
        if os.path.exists(local_model):
            model_path = local_model
        else:
            model_path = "black-forest-labs/FLUX.1-schnell"

        target_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        hf_token = os.getenv("HF_TOKEN")

        self.pipe = FluxPipeline.from_pretrained(
            model_path,
            torch_dtype=target_dtype,
            low_cpu_mem_usage=True,
            max_shard_size="5GB",
            token=hf_token,
        )

        # Performance & memory optimization combinations
        if torch.cuda.is_available():
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.device)
            self.pipe.enable_attention_slicing()

    def generate(self, prompt, steps, scale, w, h):
        self.load_model()
        optimized_prompt = rewriter.rewrite_prompt(prompt)

        # FLUX models do not use a negative prompt parameter.
        # Guidance scale for FLUX.1-schnell works best at 0.0 or low values since it's distilled.
        image = self.pipe(
            prompt=optimized_prompt,
            width=w,  # Fixed variable mapping
            height=h, # Fixed variable mapping
            guidance_scale=scale,
            num_inference_steps=steps,
            max_sequence_length=512,
            generator=torch.Generator(device=self.device).manual_seed(
                int(torch.randint(0, 999999, (1,)).item())
            )
        ).images[0]

        return image, optimized_prompt