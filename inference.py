#%%
# import os
# os.environ['CUDA_VISIBLE_DEVICES']="1"
#%%
import torch
from peft import PeftModel
import transformers
import gradio as gr
from typing import List,Union
#%%
# assert (
#     "LlamaTokenizer" in transformers._import_structure["models.llama"]
# ), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
#%%
cutoff_len=1400
BASE_MODEL = "yahma/llama-7b-hf"
tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)
# LORA_WEIGHTS = "./ckpt/generator-checkpoint-4000"
# LORA_WEIGHTS = "./run_alpaca_lora/alpaca-lora-ckpt"
# LORA_WEIGHTS = "gbharti/wealth-alpaca-lora"
# LORA_WEIGHTS = "./ckpt-full-1-1-select"
LORA_WEIGHTS = "./ckpt-multi-last"
#%%
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass
print('>>>>device:', device)
if device == "cuda":
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(model, LORA_WEIGHTS, torch_dtype=torch.float16, device_map={'': 0})
elif device == "mps":
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
else:
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL, device_map={"": device}, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        device_map={"": device},
    )
# %%
model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2
model.to(device)
#%%
"""
A dedicated helper to manage templates and prompt building.
"""
#Template
alpaca={
    "description": "Template used by Alpaca-LoRA.",
    "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
    "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
    "response_split": "### Response:"    
}

class Prompter(object):
    __slots__ = ("template", "_verbose")
    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        self.template = alpaca 
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )
    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()
# def tokenize(prompt, add_eos_token=True):
#         # there's probably a way to do this with the tokenizer settings
#         # but again, gotta move fast
#         result = tokenizer(
#             prompt,
#             truncation=True,
#             max_length=cutoff_len,
#             padding=False,
#             return_tensors=None,
#         )
#         if (
#             result["input_ids"][-1] != tokenizer.eos_token_id
#             and len(result["input_ids"]) < cutoff_len
#             and add_eos_token
#         ):
#             result["input_ids"].append(tokenizer.eos_token_id)
#             result["attention_mask"].append(1)
#         result["labels"] = result["input_ids"].copy()
#         return result
#%%
model.eval()
if torch.__version__ >= "2":
    model = torch.compile(model)
prompt_template = ""
prompter = Prompter(prompt_template)

'''
penalty_alpha=0.6
top_k=4

do_sample=False
num_beams = 1
num_beam_groups=1
top_p = 1.0
temperature = 1.0
'''
#%%
def evaluate(
    instruction,
    input=None,
    penalty_alpha = 0.6,
    do_sample=False, 
    temperature=1.0,
    top_p=1.0,
    top_k=4,
    num_beams=1,
    max_new_tokens=512,
    **kwargs,
):
    prompt = prompter.generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    # ----

    # ----
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        do_sample=do_sample,
        penalty_alpha=penalty_alpha,
        max_new_tokens=max_new_tokens,
        **kwargs,
    )
    generate_params = {
        "input_ids": input_ids,
        "generation_config": generation_config,
        "return_dict_in_generate": True,
        "output_scores": True,
        "max_new_tokens": max_new_tokens,
    }
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s, skip_special_tokens=True)
    return prompter.get_response(output)
# %%
while True:
    instruction = input("Please enter an instruction (or type 'quit' to exit):")
    user_input = input("Please enter a user prompt (or type 'quit' to exit):")
    
    if user_input.lower() == 'quit':
        break
    
    output = evaluate(instruction, user_input)
    print(f"輸出: {output}")
