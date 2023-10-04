# %%
import os
# TODO
# os.environ['CUDA_VISIBLE_DEVICES']="0"
# %%
from utils import get_parameter_names
import json
import os
import random
import sys
from typing import List, Union, Any
from collections.abc import Mapping
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import transformers
from transformers.optimization import get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.cuda.amp import autocast as autocast
import bitsandbytes as bnb
from typing import Optional, Dict, Sequence
import numpy as np
from torch.utils.data import Dataset
from dataclasses import dataclass
import torch.nn.functional as F
"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from peft.tuners.lora import LoraLayer
from transformers import (LlamaForCausalLM, LlamaTokenizer, TrainerCallback, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Seq2SeqTrainer, Seq2SeqTrainingArguments)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
# %%
# Define todo parameter
# TODO
exp_name = "multi_task"
#TODO
# Origin: 1400
cutoff_len = 1400
# TODO
# Origin: 512
rtv_cutoff_len = 512
# TODO
topk = 16
shard_size = 6
# TODO
exp_size = 100000
# TODO
test_size = 1000
# TODO
micro_batch_size = 8
# TODO
batch_size = 128
# TODO
rtv_batch_size = 16
# TODO
output_dir = './ckpt-multitask-6'
# TODO
num_epochs = 3
learning_rate = 2e-4
gradient_accumulation_steps = batch_size // micro_batch_size
# %%
data_path = "./train_test_data"
embedding_path = "./train_test_embedding"
gen_base_model = "yahma/llama-7b-hf"
rtv_base_model = "yahma/llama-30b"
rtv_lora_weight = "timdettmers/guanaco-33b"
generator_lora_weight = "tloen/alpaca-lora-7b"
DEFAULT_PAD_TOKEN = "[PAD]"
generator_prompt = 'Generate a question that a real financial analyst would ask during an earnings call conference based on the following content and the question should start with "Analyst:" : '
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# %%
def get_accelerate_model():
    device_map = "auto"
    print(f'loading base model {rtv_base_model}...')
    compute_dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        rtv_base_model,
        load_in_4bit=True,
        device_map=device_map,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
        torch_dtype=torch.bfloat16
    )

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    model.config.torch_dtype=torch.bfloat16

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable()
    model = PeftModel.from_pretrained(model, rtv_lora_weight, is_trainable=True)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    return model

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if True: trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
# %%
# Definfe model and tokenizer
# Generator
generator = LlamaForCausalLM.from_pretrained(
    gen_base_model,
    load_in_8bit=True,
    # torch_dtype=torch.float16,
    torch_dtype=torch.bfloat16,
    device_map={'':0},
)
print('>>prepare int8 model')
generator = prepare_model_for_int8_training(generator)
generator = PeftModel.from_pretrained(generator, generator_lora_weight, \
# torch_dtype=torch.float16, 
torch_dtype=torch.bfloat16,
device_map={'': 0}, is_trainable=True)
generator.config.use_cache = False
print('>>>>>Trainable generator parameter:<<<<<',) 
generator.print_trainable_parameters()
# %%
# Retriever
retriever = get_accelerate_model()
retriever.config.use_cache = False
print('>>>>>Trainable retriever parameter:<<<<<',) 
print_trainable_parameters(retriever)
# Verifying the datatypes.
dtypes = {}
for _, p in retriever.named_parameters():
    dtype = p.dtype
    if dtype not in dtypes: dtypes[dtype] = 0
    dtypes[dtype] += p.numel()
total = 0
for k, v in dtypes.items(): total+= v
for k, v in dtypes.items():
    print(k, v, v/total)
# %%
# Generator Tokenizer
gen_tokenizer = LlamaTokenizer.from_pretrained(gen_base_model)
bos = gen_tokenizer.bos_token_id
eos = gen_tokenizer.eos_token_id
pad = gen_tokenizer.pad_token_id
print("pre-trained model's BOS EOS and PAD token id:",bos,eos,pad," => It should be 1,2,none")
gen_tokenizer.pad_token_id =0 
gen_tokenizer.padding_side = "left"  # Allow batched inference
# Retriever Tokenizer
rtv_tokenizer = AutoTokenizer.from_pretrained(
        rtv_base_model,
        padding_side="left",
        use_fast=False, # Fast tokenizer giving issues.
        tokenizer_type='llama', # Needed for HF name change
    )
if rtv_tokenizer._pad_token is None:
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
        tokenizer=rtv_tokenizer,
        model=retriever,
    )
print('>>>>>Adding special tokens for retriever.')
rtv_tokenizer.add_special_tokens({
        "eos_token": rtv_tokenizer.convert_ids_to_tokens(retriever.config.eos_token_id),
        "bos_token": rtv_tokenizer.convert_ids_to_tokens(retriever.config.bos_token_id),
        "unk_token": rtv_tokenizer.convert_ids_to_tokens(
            retriever.config.pad_token_id if retriever.config.pad_token_id != -1 else rtv_tokenizer.pad_token_id
        ),
})
# %%
# Alplaca format:
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
prompter = Prompter("")
add_eos_token = True
train_on_inputs = False
def tokenize(prompt, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = gen_tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        # TODO (Origin: False)
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != gen_tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(gen_tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result

def generate_and_tokenize_prompt(data_point, cutoff_len=1400):
    # Tokenize instruction, input, and output without truncation.
    instruction_tokens = gen_tokenizer(data_point["instruction"], truncation=False)["input_ids"]
    input_tokens = gen_tokenizer(data_point["input"], truncation=False)["input_ids"]
    output_tokens = gen_tokenizer(data_point["output"], truncation=False)["input_ids"]

    # Check if the total length exceeds the maximum length.
    total_length = len(instruction_tokens) + len(input_tokens) + len(output_tokens)
    if total_length > cutoff_len:
        # Calculate the available space for input.
        available_space = cutoff_len - len(instruction_tokens) - len(output_tokens)
        
        # Truncate the input to fit the available space.
        input_tokens = input_tokens[:available_space]

        # Convert tokens back to string.
        truncated_input = gen_tokenizer.decode(input_tokens)

        # Replace the original input with the truncated input.
        data_point["input"] = truncated_input

    full_prompt = prompter.generate_prompt(
        data_point["instruction"],
        data_point["input"],
        data_point["output"],
    )
    tokenized_full_prompt = tokenize(full_prompt)
    if not train_on_inputs:
        user_prompt = prompter.generate_prompt(
            data_point["instruction"], data_point["input"]
        )
        tokenized_user_prompt = tokenize(
            user_prompt, add_eos_token=add_eos_token
        )
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        if add_eos_token:
            user_prompt_len -= 1

        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]  # could be sped up, probably
    return tokenized_full_prompt
# %%
# Load data
# def search_by_embedding(query, qid, emb_data):
#     assert query['fullQuestionContent'] == emb_data['questions']['sentences'][qid]['fullQuestionContent']
#     user_embed = emb_data['questions']['embs'][qid]
#     user_embed = np.array(user_embed).astype(float)

#     # use cosine similarity to find the most similar sentence from test_X
#     embd = np.array([np.array(e).astype(float) for e in emb_data['presentations']['embs']])
#     cosine_similarity = np.dot(embd, user_embed) / (np.linalg.norm(embd, axis=1) * np.linalg.norm(user_embed))
#     cosine_similarity = cosine_similarity.reshape(-1, 1)

#     # find the five most similar sentence
#     topk_idx = np.argsort(cosine_similarity, axis=0)[-topk:]
#     topk_idx = topk_idx.reshape(-1, 1)
#     topk_idx = sorted(topk_idx[::-1])

#     result = []
#     for i in topk_idx:
#         result.append(emb_data["presentations"]['sentences'][i[0]])
#     return result
from rank_bm25 import BM25Okapi
import nltk
nltk.download('punkt')
def search_by_bm25(query, text_list):
    tokenized_text_list = [nltk.word_tokenize(doc) for doc in text_list]
    bm25 = BM25Okapi(tokenized_text_list)
    tokenized_query = nltk.word_tokenize(query)
    doc_scores = bm25.get_scores(tokenized_query)
    top_n_indices = np.argsort(doc_scores)[::-1][:64]
    paragraph = " ".join([text_list[idx] for idx in sorted(top_n_indices)])
    return paragraph
def load_train_test_dataset(data_path):
    train_data = []
    test_data = []
    for file in tqdm(os.listdir(f"{data_path}/train")):
        with open(f'{data_path}/train/{file}', encoding='utf-8') as f:
            data = json.load(f)
        presentations = []
        for presentation in data['presentation']:
            if len(presentation.replace(" ", "")) > 0:
                presentations.append(presentation)
        if len(presentations) >= topk:
            for qid, question in enumerate(data['questions']):
                # TODO
                if question['label'] == 1 and len(question['keywords']) > 0:
                    # TODO 先用 embedding 找出
                    new_data = {}
                    new_data['presentations'] = search_by_bm25(question['fullQuestionContent'], presentations)
                    new_data['question'] = question['fullQuestionContent']
                    # new_data['keywords'] = question['keywords']
                    train_data.append(new_data)
                    
    for file in tqdm(os.listdir(f"{data_path}/test")):
        with open(f'{data_path}/test/{file}', encoding='utf-8') as f:
            data = json.load(f)
        presentations = []
        for presentation in data['presentation']:
            if len(presentation.replace(" ", "")) > 0:
                presentations.append(presentation)
        if len(presentations) >= topk:
            for qid, question in enumerate(data['questions']):
                # TODO
                if question['label'] == 1 and len(question['keywords']) > 0:
                    # TODO 先用 embedding 找出
                    new_data = {}
                    new_data['presentations'] = search_by_bm25(question['fullQuestionContent'], presentations)
                    new_data['question'] = question['fullQuestionContent']
                    # new_data['keywords'] = question['keywords']
                    test_data.append(new_data)
    return train_data, test_data
# %%
train_data, test_data = load_train_test_dataset(data_path)
# with open(f'/root/autodl-tmp/remote_data/train.json', encoding='utf-8') as f:
#     train_data = json.load(f)
# with open(f'/root/autodl-tmp/remote_data/test.json', encoding='utf-8') as f:
#     test_data = json.load(f)
# 打亂 train_data
random.shuffle(train_data)
random.shuffle(test_data)
train_data = train_data[:exp_size]
test_data = test_data[:test_size]
# %%
class ConcallDataset(Dataset):
    def __init__(self, data: list):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx]['question']
        question_text = "Analyst:" + " " + question.strip()
        # Process presentation
        presentation_list = self.data[idx]['presentations']
        
        return question_text, list(presentation_list)
# %%
train_dataset = ConcallDataset(train_data)
test_dataset = ConcallDataset(test_data)
print('>>>len of train_data:', len(train_dataset), 'len of test data:', len(test_dataset))
# TODO
gen_num_sample=3 #Randmly pick 3 instances from val_dataset
gen_dataset = random.sample(list(test_dataset), gen_num_sample)
# %%
@dataclass
class MyDataCollator:
    def __call__(self, batch: Sequence[Dict]) -> Dict[str, List]:
        questions, presentations = zip(*batch)
        return {"batch": {'question': list(questions), 'presentations': list(presentations)}}

# %%
class MultiTaskModel(transformers.PreTrainedModel):
    def __init__(self, rtv_tokenizer, gen_tokenizer, retriever, generator):
        # super().__init__()
        super().__init__(transformers.PretrainedConfig())
        self.rtv_tokenizer = rtv_tokenizer
        self.gen_tokenizer = gen_tokenizer
        self.retriever = retriever
        self.generator = generator
        # self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.highly_id = self.rtv_tokenizer.encode('Highly')[1:-1]
        self.partially_id_1, self.partially_id_2 = self.rtv_tokenizer.encode('Partially')[1:-1]
        self.not_id = self.rtv_tokenizer.encode('Not')[1]  # assume 'Not' is not split into subtokens
        self.shard_size = shard_size

    def save_pretrained(self, save_directory, state_dict=None, safe_serialization=False):
        self.generator.save_pretrained(os.path.join(save_directory, "generator"))
        self.retriever.save_pretrained(os.path.join(save_directory, "retriever"))


    def _find_top_shard_size(self, rtv_all_batch_not_idx, rtv_all_batch_scores):
        batch_shape = rtv_all_batch_not_idx.shape[0]
        not_count = rtv_all_batch_not_idx.sum(dim=1)  # 计算每个 batch 中 'Not' 的数量
        remaining_count = rtv_all_batch_scores.shape[1] - not_count  # 计算每个 batch 中剩余的 '非Not' 项数
        dynamic_shard_size = torch.min(self.shard_size * torch.ones_like(remaining_count), remaining_count)  # 取 shard_size 和 remaining_count 的最小值
        student_top_shard_size_values, student_top_shard_size_indices = [None]*batch_shape, [None]*batch_shape
        for i in range(batch_shape):
            if dynamic_shard_size[i].item() > 0:
                top_values, top_indices = torch.topk(rtv_all_batch_scores[i], k=int(dynamic_shard_size[i].item()))
            else:  # 如果去掉那些Not之后就没有剩下了，那就按照分数取前 shard_size 高分的即可
                top_values, top_indices = torch.topk(rtv_all_batch_scores[i], k=self.shard_size)
            # pad top_indices to be of shape [shard_size]
            top_indices_padded = torch.full((self.shard_size,), -1).to(top_indices.device)  # create a tensor of -1s with shape [shard_size]
            top_indices_padded[:len(top_indices)] = top_indices  # replace the beginning of this tensor with the top_indices
            student_top_shard_size_values[i] = top_values
            student_top_shard_size_indices[i] = top_indices_padded
        return student_top_shard_size_values, torch.stack(student_top_shard_size_indices)


    def _pad_gen_batch_data(self, features, tokenizer=gen_tokenizer, padding=True, max_length=cutoff_len, pad_to_multiple_of=8, return_tensors='pt'):
        label_pad_token_id = -100
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + pad_to_multiple_of - 1)
                    // pad_to_multiple_of
                    * pad_to_multiple_of
                )

            padding_side = tokenizer.padding_side
            for feature in features:
                remainder = [label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        features = tokenizer.pad(
            features,
            padding=padding,
            max_length=None,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        return features

    def _postprocess(self, batch_questions, batch_presentations, batch_top_shard_idx):
        data_points = []
        for bid in range(batch_top_shard_idx.shape[0]): # batch size
            # Sorted version
            # valid_indices = [i for i in sorted(batch_top_shard_idx[bid]) if i != -1]
            # Not sorted version
            valid_indices = [i for i in batch_top_shard_idx[bid] if i != -1]
            selected_presentation = " ".join([batch_presentations[bid][i] for i in valid_indices])
            question = batch_questions[bid]
            data_points.append({
                "instruction": generator_prompt,
                "input": selected_presentation,
                "output": question
            })
            # print('>>data:', {
            #     "instruction": generator_prompt,
            #     "input": selected_presentation,
            #     "output": question
            # })
        batch_data = [generate_and_tokenize_prompt(data, cutoff_len) for data in data_points]
        # 從 batch_data 中提取 input_ids、labels 和 attention_mask，並將它們堆疊成張量
        batch_data = self._pad_gen_batch_data(batch_data)
        #這邊要把資料 pad 到同樣的長度
        input_ids = batch_data['input_ids']
        labels = batch_data['labels']
        attention_mask = batch_data['attention_mask']
        return input_ids, attention_mask, labels

    def forward(self, batch):
        assert len(batch['question']) == len(batch['presentations'])
        batch_shape = len(batch['question'])
        batch_questions = batch['question']
        batch_presentations = batch['presentations']
        num_presentations = len(batch_presentations[0])
        assert num_presentations == topk
        rtv_all_batch_scores = torch.zeros((batch_shape, num_presentations))  # num_presentations 是 presentations 的数量
        rtv_all_batch_not_idx = torch.zeros((batch_shape, num_presentations))

        for bid in range(batch_shape):
            presentations = batch_presentations[bid]
            question = batch_questions[bid]
            rtv_prompts = [f'### Human: Given the transcript of a presentation by a manager during an earnings call conference and a question raised by a financial analyst, is the question highly related, partially related, or not related to the information provided by the manager? ("Highly Related"/"Partially Related"/"Not Related") Transcript: {presentation}  Question: {question} ### Assistant: The answer is \n\n"' for presentation in presentations]
            tokenized_prompt = self.rtv_tokenizer(rtv_prompts, truncation=True, max_length=rtv_cutoff_len, padding=True, return_tensors="pt")
            tokenized_prompt = {k: v.to(self.device) for k, v in tokenized_prompt.items()}
            for i in range(0, len(presentations), rtv_batch_size):
                rtv_predictions = self.retriever(input_ids = tokenized_prompt['input_ids'][i:i+rtv_batch_size], attention_mask=tokenized_prompt['attention_mask'][i:i+rtv_batch_size])[0]
                rtv_probs = torch.nn.functional.softmax(rtv_predictions[:, -1, :], dim=-1)
                # 計算 Highly + Partially - Not 的機率
                highly_probs = rtv_probs.index_select(1, torch.tensor(self.highly_id).to(rtv_probs.device)).squeeze()
                partially_probs_1 = rtv_probs.index_select(1, torch.tensor([self.partially_id_1]).to(rtv_probs.device)).squeeze()
                partially_probs_2 = rtv_probs.index_select(1, torch.tensor([self.partially_id_2]).to(rtv_probs.device)).squeeze()
                not_probs = rtv_probs.index_select(1, torch.tensor([self.not_id]).to(rtv_probs.device)).squeeze()
                # 计算"Highly + Partially - Not"的分数
                rtv_scores = (highly_probs) + (partially_probs_1 + partially_probs_2)/2 - not_probs
                rtv_all_batch_scores[bid, i:i+rtv_batch_size] = rtv_scores
                # 找到最大生成機率不是 Not 的那些 presentation (如果是 Not -> 0; 不是 Not -> 1)
                _, max_prob_ids = torch.max(rtv_probs, dim=-1)
                not_idx_result = (max_prob_ids == self.not_id).int()
                rtv_all_batch_not_idx[bid, i:i+rtv_batch_size] = not_idx_result
        # student_probs = F.log_softmax(rtv_all_batch_scores, dim=1)
        student_all_batch_top_shard_size_values, student_all_batch_top_shard_size_indices = self._find_top_shard_size(rtv_all_batch_not_idx, rtv_all_batch_scores)
        # 只有 training mode 之下，需要計算 teacher distribution，目的是為了更新 retriever
        if self.training:
            pass
        # 訓練 Generator
        input_ids, attention_mask, label = self._postprocess(batch_questions, batch_presentations,  student_all_batch_top_shard_size_indices)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        label = label.to(self.device)
        lm_outputs = self.generator(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=label
                    )
        loss = lm_outputs.loss
        return {"loss": loss}
# %%
model = MultiTaskModel(rtv_tokenizer=rtv_tokenizer, gen_tokenizer=gen_tokenizer, generator=generator, retriever=retriever)
# if torch.__version__ >= "2" and sys.platform != "win32":
#     model = torch.compile(model)
# %%
# 處理qlora, alpaca-lora的其他要傳入model的參數
from os.path import exists, join, isdir
class SavePeftModelCallback(TrainerCallback):
    def save_model(self, args, state, kwargs):
        print('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        model = kwargs["model"]

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

        if isinstance(model, MultiTaskModel):
            generator_folder = os.path.join(checkpoint_folder, "generator")
            retriever_folder = os.path.join(checkpoint_folder, "retriever")

            model.generator.save_pretrained(generator_folder)
            model.retriever.save_pretrained(retriever_folder)
        else:
            peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
            model.save_pretrained(peft_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)
        touch(join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)
# %%
class GenerateTextCallback(TrainerCallback):
    def __init__(self,generator, retriever, rtv_tokenizer, gen_tokenizer, device, gen_dataset, max_length, shard_size): 
        self.generator = generator
        self.retriever = retriever
        self.rtv_tokenizer = rtv_tokenizer
        self.gen_tokenizer = gen_tokenizer
        self.device = device
        self.gen_dataset=gen_dataset 
        self.max_length = max_length
        self.highly_id = self.rtv_tokenizer.encode('Highly')[1:-1]
        self.partially_id_1, self.partially_id_2 = self.rtv_tokenizer.encode('Partially')[1:-1]
        self.not_id = self.rtv_tokenizer.encode('Not')[1]
        self.shard_size = shard_size

    def generate_text(self,prompt):
        self.generator.eval()
        # Generate text
        self.gen_tokenizer.padding_side = "left"
        self.gen_tokenizer.pad_token_id = 0
        input_ids =self.gen_tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated_ids = self.generator.generate(
            input_ids=input_ids,
            max_new_tokens=256,
            bos_token_id=1,
            eos_token_id=2,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            num_beams=1,
            num_return_sequences=1
            )
        output = self.gen_tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        return output
    def gather_input(self, batch_presentations, batch_questions, batch_top_shard_idx):
        data_points = []
        bid = 0
        valid_indices = [i for i in batch_top_shard_idx[bid] if i != -1]
        selected_presentation = " ".join([batch_presentations[bid][i] for i in valid_indices])
        question = batch_questions[bid]
        data_points.append({
            "instruction": generator_prompt,
            "input": selected_presentation,
            "output": question
        })
        return data_points
    def _find_top_shard_size(self, rtv_all_batch_not_idx, rtv_all_batch_scores):
        batch_shape = rtv_all_batch_not_idx.shape[0]
        not_count = rtv_all_batch_not_idx.sum(dim=1)  # 计算每个 batch 中 'Not' 的数量
        remaining_count = rtv_all_batch_scores.shape[1] - not_count  # 计算每个 batch 中剩余的 '非Not' 项数
        dynamic_shard_size = torch.min(self.shard_size * torch.ones_like(remaining_count), remaining_count)
  # 取 shard_size 和 remaining_count 的最小值
        student_top_shard_size_values, student_top_shard_size_indices = [None]*batch_shape, [None]*batch_shape
        for i in range(batch_shape):
            if dynamic_shard_size[i].item() > 0:
                top_values, top_indices = torch.topk(rtv_all_batch_scores[i], k=int(dynamic_shard_size[i].item()))
            else:  # 如果去掉那些Not之后就没有剩下了，那就按照分数取前 shard_size 高分的即可
                top_values, top_indices = torch.topk(rtv_all_batch_scores[i], k=self.shard_size)
            # pad top_indices to be of shape [shard_size]
            top_indices_padded = torch.full((self.shard_size,), -1).to(top_indices.device)  # create a tensor of -1s with shape [shard_size]
            top_indices_padded[:len(top_indices)] = top_indices  # replace the beginning of this tensor with the top_indices
            student_top_shard_size_values[i] = top_values
            student_top_shard_size_indices[i] = top_indices_padded
        return student_top_shard_size_values, torch.stack(student_top_shard_size_indices)

    def on_evaluate(self, args, state, control, **kwargs):
        self.retriever.eval()
        # question, presentation_list
        for id in range(len(self.gen_dataset)):
            presentations = self.gen_dataset[id][1]
            question = self.gen_dataset[id][0]
            rtv_prompts = [f'### Human: Given the transcript of a presentation by a manager during an earnings call conference and a question raised by a financial analyst, is the question highly related, partially related, or not related to the information provided by the manager? ("Highly Related"/"Partially Related"/"Not Related") Transcript: {presentation}  Question: {question} ### Assistant: The answer is \n\n"' for presentation in presentations]
            tokenized_prompt = self.rtv_tokenizer(rtv_prompts, truncation=True, max_length=rtv_cutoff_len, padding=True, return_tensors="pt")
            tokenized_prompt = {k: v.to(self.device) for k, v in tokenized_prompt.items()}
            rtv_all_batch_scores = torch.zeros((1, len(presentations)))  # len(presentations) 是 presentations 的数量
            rtv_all_batch_not_idx = torch.zeros((1, len(presentations)))
            bid = 0
            for pid in range(0, len(presentations), rtv_batch_size):
                with torch.no_grad():
                    rtv_predictions = self.retriever(input_ids = tokenized_prompt['input_ids'][pid:pid+rtv_batch_size], attention_mask = tokenized_prompt['attention_mask'][pid:pid+rtv_batch_size])[0]
                rtv_probs = torch.nn.functional.softmax(rtv_predictions[:, -1, :], dim=-1)
                # 計算 Highly + Partially - Not 的機率
                highly_probs = rtv_probs.index_select(1, torch.tensor(self.highly_id).to(rtv_probs.device)).squeeze()
                # highly_probs_2 = rtv_probs.index_select(1, torch.tensor([self.highly_id_2]).to(rtv_probs.device)).squeeze()
                partially_probs_1 = rtv_probs.index_select(1, torch.tensor([self.partially_id_1]).to(rtv_probs.device)).squeeze()
                partially_probs_2 = rtv_probs.index_select(1, torch.tensor([self.partially_id_2]).to(rtv_probs.device)).squeeze()
                not_probs = rtv_probs.index_select(1, torch.tensor([self.not_id]).to(rtv_probs.device)).squeeze()
                # 计算"Highly + Partially - Not"的分数
                rtv_scores = highly_probs + (partially_probs_1 + partially_probs_2)/2 - not_probs
                rtv_all_batch_scores[bid, pid:pid+rtv_batch_size] = rtv_scores
                # 找到最大生成機率不是 Not 的那些 presentation (如果是 Not -> 0; 不是 Not -> 1)
                _, max_prob_ids = torch.max(rtv_probs, dim=-1)
                not_idx_result = (max_prob_ids == self.not_id).int()
                rtv_all_batch_not_idx[bid, pid:pid+rtv_batch_size] = not_idx_result
        # student_probs = F.log_softmax(rtv_all_batch_scores, dim=1)
            student_all_batch_top_shard_size_values, student_all_batch_top_shard_size_indices = self._find_top_shard_size(rtv_all_batch_not_idx, rtv_all_batch_scores)

            data = self.gather_input([presentations], [question], student_all_batch_top_shard_size_indices)

            prompt = alpaca["prompt_input"].format(
                instruction=data[0]['instruction'], input=data[0]['input']
            )
            #print("prompt:",prompt)
            generated_text = self.generate_text(prompt)
            print(f"\nSample {id+1}:\n Instruction: {data[0]['instruction']}\n Input: {data[0]['input']}\n Output:{data[0]['output']}\n\n Predict:\n {generated_text} \n=> The correct answer should follow the aplaca template.\n")
# %%
generate_text_callback = GenerateTextCallback(generator=generator, retriever=retriever, rtv_tokenizer=rtv_tokenizer, gen_tokenizer=gen_tokenizer, device=device, gen_dataset=gen_dataset, max_length=cutoff_len, shard_size = shard_size)
# %%
# generate_text_callback.on_evaluate(1,2,3)
# %%
#TrainingArguments
#Seq2SeqTrainingArguments
import logging
logging.basicConfig(level=logging.INFO)
training_args = Seq2SeqTrainingArguments(
    do_train = True,
    do_eval= True,
    per_device_train_batch_size=micro_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    warmup_steps=30,
    num_train_epochs=num_epochs,
    learning_rate=learning_rate,
    bf16=True,
    # TODO
    logging_steps=2,
    optim="paged_adamw_32bit",
    lr_scheduler_type="linear",
    evaluation_strategy="steps" if test_size > 0 else "no",
    save_strategy="steps",
    # TODO
    eval_steps=48 if test_size > 0 else None,
    # TODO
    save_steps=48,
    output_dir=output_dir,
    save_total_limit=3,
    group_by_length=False,
    log_level="info",
    # gradient_checkpointing=True,
    # max_grad_norm=0.3,
)
trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,           # training dataset
        eval_dataset=test_dataset,
        data_collator=MyDataCollator(),
        # callbacks=[SavePeftModelCallback, generate_text_callback],
        callbacks=[SavePeftModelCallback(), generate_text_callback],
    )
# trainer.add_callback(SavePeftModelCallback)
# %%
train_result = trainer.train()
# %%
os.makedirs("./ckpt-multi-last", exist_ok=True)
os.makedirs("./ckpt-multi-last/generator", exist_ok=True)
os.makedirs("./ckpt-multi-last/retriever", exist_ok=True)
model.generator.save_pretrained("./ckpt-multi-last/generator")
model.retriever.save_pretrained("./ckpt-multi-last/retriever")