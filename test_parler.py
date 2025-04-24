import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, set_seed
import soundfile as sf

device = "cuda:0" if torch.cuda.is_available() else "cpu"

attn_implementation = "eager" # "eager", "sdpa" or "flash_attention_2"

model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-expresso", attn_implementation=attn_implementation).to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-expresso", use_fast=False)

prompt = "I took the one less traveled by, and that has made all the difference."
description = "Jerry speaks in a disgusted tone with clear articulation."

input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

set_seed(42)
generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids, bos_token_id=1025, decoder_start_token_id=1025, do_sample=True, eos_token_id=1024, max_new_tokens=2580, min_new_tokens=10, pad_token_id=1024)

audio_arr = generation.cpu().numpy().squeeze()
sf.write("parler_tts_out.wav", audio_arr, model.config.sampling_rate)
