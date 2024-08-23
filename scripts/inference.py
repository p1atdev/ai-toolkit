import torch
import gc
import os

from optimum.quanto import freeze, qfloat8, quantize

from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from transformers import CLIPTextModel, CLIPTokenizer,T5EncoderModel, T5TokenizerFast

from toolkit.sampler import get_sampler

def flush():
    torch.cuda.empty_cache()
    gc.collect()

model_path = "black-forest-labs/FLUX.1-dev"
device_torch = torch.device("cuda")
dtype = torch.float16

sampler = get_sampler(
    "flowmatch",
    {
        "prediction_type": "epsilon",
    },
    'sd'
)

print("Loading Flux model")
base_model_path = "black-forest-labs/FLUX.1-schnell"
print("Loading transformer")
subfolder = 'transformer'
transformer_path = model_path
local_files_only = False
# check if HF_DATASETS_OFFLINE or TRANSFORMERS_OFFLINE is set
if os.path.exists(transformer_path):
    subfolder = None
    transformer_path = os.path.join(transformer_path, 'transformer')
    # check if the path is a full checkpoint.
    te_folder_path = os.path.join(model_path, 'text_encoder')
    # if we have the te, this folder is a full checkpoint, use it as the base
    if os.path.exists(te_folder_path):
        base_model_path = model_path

transformer = FluxTransformer2DModel.from_pretrained(
    transformer_path,
    subfolder=subfolder,
    torch_dtype=dtype,
    # low_cpu_mem_usage=False,
    # device_map=None
)
# for low v ram, we leave it on the cpu. Quantizes slower, but allows training on primary gpu
# TODO: EXPERIMENT CPU AND GPU HERE
transformer.to(torch.device("cpu"), dtype=dtype)
flush()

# need the pipe to do this unfortunately for now
# we have to fuse in the weights before quantizing
pipe: FluxPipeline = FluxPipeline(
    scheduler=None,
    text_encoder=None,
    tokenizer=None,
    text_encoder_2=None,
    tokenizer_2=None,
    vae=None,
    transformer=transformer,
)
pipe.load_lora_weights("/home/shai.kadish/models/lora/flux_anthelios/ilias_flux_A100/my_first_flux_lora_anpd001_v5_000002500.safetensors", adapter_name="lora1")
pipe.fuse_lora()
# unfortunately, not an easier way with peft
pipe.unload_lora_weights()

quantization_type = qfloat8
print("Quantizing transformer")
quantize(transformer, weights=quantization_type)
freeze(transformer)
transformer.to(device_torch)

flush()

scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(base_model_path, subfolder="scheduler")
print("Loading vae")
vae = AutoencoderKL.from_pretrained(base_model_path, subfolder="vae", torch_dtype=dtype)
flush()

print("Loading t5")
tokenizer_2 = T5TokenizerFast.from_pretrained(base_model_path, subfolder="tokenizer_2", torch_dtype=dtype)
text_encoder_2 = T5EncoderModel.from_pretrained(base_model_path, subfolder="text_encoder_2",
                                                torch_dtype=dtype)

text_encoder_2.to(device_torch, dtype=dtype)
flush()

print("Quantizing T5")
quantize(text_encoder_2, weights=qfloat8)
freeze(text_encoder_2)
flush()

print("Loading clip")
text_encoder = CLIPTextModel.from_pretrained(base_model_path, subfolder="text_encoder", torch_dtype=dtype)
tokenizer = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer", torch_dtype=dtype)
text_encoder.to(device_torch, dtype=dtype)

print("making pipe")
pipe: FluxPipeline = FluxPipeline(
    scheduler=scheduler,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    text_encoder_2=None,
    tokenizer_2=tokenizer_2,
    vae=vae,
    transformer=None,
)
pipe.text_encoder_2 = text_encoder_2
pipe.transformer = transformer

print("preparing")

text_encoder = [pipe.text_encoder, pipe.text_encoder_2]
tokenizer = [pipe.tokenizer, pipe.tokenizer_2]

pipe.transformer = pipe.transformer.to(device_torch)

flush()
text_encoder[0].to(device_torch)
text_encoder[0].requires_grad_(False)
text_encoder[0].eval()
text_encoder[1].to(device_torch)
text_encoder[1].requires_grad_(False)
text_encoder[1].eval()
pipe.transformer = pipe.transformer.to(device_torch)
flush()
pipe = pipe.to("cuda")
pipe.scheduler = sampler
