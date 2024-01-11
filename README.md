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
from llama_index_extra_llm.deepseek import DeepSeekLLM
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
service_context = ServiceContext.from_defaults(llm=llm, embed_model="local:BAAI/bge-small-en-v1.5")
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents, service_context=service_context)
query_engine = index.as_query_engine(streaming=True, similarity_top_k=1)
prompt=DeepSeekLLM.messages2prompt(messages=[{"role": "user", "content": "Hello"}])
streaming_response=query_engine.query(prompt)
streaming_response.print_response_stream()
```