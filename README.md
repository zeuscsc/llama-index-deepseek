# LlamaIndex Extra LLM
Just a simple extension for LlamaIndex for better apply some llm such as DeepSeek.

## Features
- [x] Support DeepSeek

## Installation / Environment
Pytorch is needed, it is easier to install by conda if you are using local PC with GPU
```shell
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

## Quick Usage
Quantization is optional
```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
llm = DeepSeekLLM(
    model_name="deepseek-ai/deepseek-llm-7b-chat",
    tokenizer_name="deepseek-ai/deepseek-llm-7b-chat",
    context_window=3900,
    max_new_tokens=1024,
    model_kwargs={"quantization_config": quantization_config},
    generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    device_map="auto",
)
```