import json
from pathlib import Path
import gc
from PIL import Image

import torch
from safetensors.torch import load_file 
from optimum.quanto import freeze, qfloat8, quantize, requantize
from diffusers import AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from transformers import CLIPTextModel, CLIPTokenizer,T5EncoderModel, T5TokenizerFast

from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler
from .save_quantized_transformer import QUANTIZED_TRANSFORMER_NAME, TRANSFORMER_QUANTIZATION_MAP_NAME

def flush():
    torch.cuda.empty_cache()
    gc.collect()

def load_transformer(quantized_transformer_dir:Path|str) -> FluxTransformer2DModel:
    quantized_transformer_dir = Path(quantized_transformer_dir)
    state_dict = load_file(quantized_transformer_dir/QUANTIZED_TRANSFORMER_NAME)

    with open(quantized_transformer_dir/TRANSFORMER_QUANTIZATION_MAP_NAME, "r") as f:
        quantization_map = json.load(f)

    with torch.device('meta'):
        transformer = FluxTransformer2DModel()
    
    requantize(transformer, state_dict, quantization_map, "cpu")
    return transformer

class FluxModel:
    def __init__(self, transformer: FluxTransformer2DModel, base_model_name: str = "black-forest-labs/FLUX.1-dev"):
        self.device = torch.device("cuda")
        self.dtype = torch.float16

        scheduler = self.load_scheduler()
        vae = AutoencoderKL.from_pretrained(base_model_name, subfolder="vae", torch_dtype=self.dtype)
        tokenizer, tokenizer_2, text_encoder, text_encoder_2 = self.load_text_processing_components(base_model_name)

        self.load_pipe(scheduler, vae, tokenizer, tokenizer_2, text_encoder, text_encoder_2)
        self.switch_transformer(transformer)

    def __call__(self, prompt: str, inference_args: dict, save_as: Path | None | str = None) -> Image:
        with torch.cuda.amp.autocast(): 
            out_image = self.pipe(prompt, **inference_args).images[0]
        
        if save_as is not None:
            out_image.save(save_as)
        
        return out_image


    def load_scheduler(self) -> CustomFlowMatchEulerDiscreteScheduler:
            config = {
                "_class_name": "FlowMatchEulerDiscreteScheduler",
                "_diffusers_version": "0.29.0.dev0",
                "num_train_timesteps": 1000,
                "shift": 3.0
            }
            return CustomFlowMatchEulerDiscreteScheduler.from_config(config)


    def load_text_processing_components(self, model_name: str) -> tuple[CLIPTokenizer, T5TokenizerFast, CLIPTextModel, T5EncoderModel]:
        tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer", torch_dtype=self.dtype)
        tokenizer_2 = T5TokenizerFast.from_pretrained(model_name, subfolder="tokenizer_2", torch_dtype=self.dtype)
        
        text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder", torch_dtype=self.dtype)
        text_encoder.to(self.device, dtype=self.dtype)
        flush()

        text_encoder_2 = T5EncoderModel.from_pretrained(model_name, subfolder="text_encoder_2", torch_dtype=self.dtype)
        text_encoder_2.to(self.device, dtype=self.dtype)
        flush()
        quantize(text_encoder_2, weights=qfloat8)
        freeze(text_encoder_2)

        return tokenizer, tokenizer_2, text_encoder, text_encoder_2


    def load_pipe(self, scheduler: CustomFlowMatchEulerDiscreteScheduler, vae: AutoencoderKL, tokenizer: CLIPTokenizer, tokenizer_2: T5TokenizerFast, text_encoder: CLIPTextModel, text_encoder_2: T5EncoderModel):
        pipe = FluxPipeline(
            scheduler=scheduler,
            text_encoder=text_encoder,
            text_encoder_2 = text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            vae=vae,
            transformer=None,
        )
        self.pipe = pipe.to(self.device)
        flush()
        
   def switch_transformer(self, transformer: FluxTransformer2DModel):
        if self.pipe.transformer:
            self.pipe.transformer.to("cpu")
        self.pipe.transformer = transformer.to(self.device)
