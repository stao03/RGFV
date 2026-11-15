# -*- coding: utf-8 -*-
"""
local_llm.py

本地加载 Qwen2.5-7B-Instruct，提供：
1. generate(prompt): 用于情感分类生成
2. embed(texts): 用于提取文本 embedding，给补全检索使用

依赖:
pip install transformers torch accelerate
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = 'Llama-3.1-8B-Instruct'

_tok = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

if _tok.pad_token is None:
    _tok.pad_token = _tok.eos_token

_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map='auto',
    trust_remote_code=True
)

_model.eval()

SYSTEM = "You are an expert in the field of sentiment analysis."


def _get_input_device():
    """
    device_map='auto' 时，模型可能分布在多张卡上。
    输入一般放到第一个参数所在的 device。
    """
    try:
        return _model.device
    except Exception:
        return next(_model.parameters()).device


@torch.no_grad()
def generate(prompt, max_new_tokens=200):
    msgs = [
        {'role': 'system', 'content': SYSTEM},
        {'role': 'user', 'content': prompt}
    ]

    text = _tok.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True
    )

    device = _get_input_device()

    inputs = _tok(
        [text],
        return_tensors='pt'
    ).to(device)

    out = _model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        top_k=None,
        top_p=None,
        temperature=None,
        pad_token_id=_tok.eos_token_id
    )

    gen = out[0][inputs.input_ids.shape[1]:]

    return _tok.decode(gen, skip_special_tokens=True)


@torch.no_grad()
def embed(texts, batch_size=4, max_length=256):
    """
    使用当前 Qwen2.5-7B-Instruct 提取文本向量。

    参数:
        texts:
            str 或 list[str]
        batch_size:
            Qwen2.5-7B 比较吃显存，建议先用 2 或 4。
        max_length:
            CMU-MOSI utterance 一般不长，256 足够。

    返回:
        如果输入 str:
            np.ndarray, shape = [hidden_dim]
        如果输入 list[str]:
            np.ndarray, shape = [n, hidden_dim]
    """
    single = False

    if isinstance(texts, str):
        texts = [texts]
        single = True

    texts = [str(t) if t is not None else '' for t in texts]

    all_vecs = []
    device = _get_input_device()

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        inputs = _tok(
            batch,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)

        outputs = _model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )

        # [B, L, H]
        hidden = outputs.hidden_states[-1]

        # [B, L, 1]
        mask = inputs.attention_mask.unsqueeze(-1).to(hidden.device).float()

        # mean pooling，只平均真实 token，不平均 padding
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

        # L2 normalize，后面可以直接用点积算 cosine similarity
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

        all_vecs.append(pooled.float().cpu())

    vecs = torch.cat(all_vecs, dim=0).numpy()

    if single:
        return vecs[0]

    return vecs