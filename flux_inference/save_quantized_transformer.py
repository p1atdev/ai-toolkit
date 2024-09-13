import json
from pathlib import Path
from argparse import ArgumentParser

import torch
from optimum.quanto import freeze, qfloat8, quantize, quantization_map
from safetensors.torch import save_file
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline

QUANTIZED_TRANSFORMER_NAME = 'quantized_transformer.safetensors'
TRANSFORMER_QUANTIZATION_MAP_NAME = 'transformer_quantization_map.json'

def save_quantized_flux_transformer(output_path: str|Path, base_model: str|Path = "black-forest-labs/FLUX.1-dev", lora_path: str|Path|None = None):
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)
    dtype = torch.float16

    transformer = FluxTransformer2DModel.from_pretrained(
        base_model,
        subfolder="transformer",
        torch_dtype=dtype,
    )
    transformer.to(torch.device("cpu"), dtype=dtype)

    if lora_path:
        # Must load Lora to pipe to fuse transformer with Lora in-place
        pipe = FluxPipeline(
            scheduler=None,
            text_encoder=None,
            tokenizer=None,
            text_encoder_2=None,
            tokenizer_2=None,
            vae=None,
            transformer=transformer,
        )
        pipe.load_lora_weights(lora_path)
        pipe.fuse_lora()
        pipe.unload_lora_weights()

    quantize(transformer, weights=qfloat8)
    freeze(transformer)

    save_file(transformer.state_dict(), output_path/QUANTIZED_TRANSFORMER_NAME)
    with open(output_path/TRANSFORMER_QUANTIZATION_MAP_NAME, "w") as f:
        json.dump(quantization_map(transformer), f)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("output_path", type=Path, help="Path to output directory for quantized model.")
    parser.add_argument("--base_model", type=str, help="Name of flux base model to load from huggingface, or path to local model location.", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--lora_path", type=Path, help="If set, quantize transformer with lora model loaded.")
    args = parser.parse_args()

    save_quantized_flux_transformer(args.output_path, args.base_model, args.lora_path)
