import torch
from transformers import BitsAndBytesConfig
from llama_index.prompts import PromptTemplate
from llama_index.llms import HuggingFaceLLM

from typing import Optional, List, Any, Union, Callable

from llama_index.callbacks import CallbackManager
from llama_index.llms import (
    CompletionResponse,
    CompletionResponseGen,
)
from llama_index.llms.base import llm_completion_callback
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig,TextIteratorStreamer
from llama_index.constants import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_NUM_OUTPUTS,
)
from threading import Thread

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

DEFAULT_HUGGINGFACE_MODEL = "deepseek-ai/deepseek-llm-7b-chat"

class DeepSeekLLM(HuggingFaceLLM):
    def messages2prompt(messages: list) -> str:
        prompt = ""
        for message in messages:
            if message["role"] == "user":
                prompt += f"""User: {message["content"]}\nAssistant: """
            else:
                prompt += f"""Assistant: {messages[1]['content']}<｜end▁of▁sentence｜>"""
        return prompt

    def __init__(
        self,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        max_new_tokens: int = DEFAULT_NUM_OUTPUTS,
        system_prompt: str = "",
        query_wrapper_prompt: Union[str, PromptTemplate] = "{query_str}",
        tokenizer_name: str = DEFAULT_HUGGINGFACE_MODEL,
        model_name: str = DEFAULT_HUGGINGFACE_MODEL,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        device_map: Optional[str] = "auto",
        stopping_ids: Optional[List[int]] = None,
        tokenizer_kwargs: Optional[dict] = None,
        tokenizer_outputs_to_remove: Optional[list] = None,
        model_kwargs: Optional[dict] = None,
        generate_kwargs: Optional[dict] = None,
        is_chat_model: Optional[bool] = False,
        messages_to_prompt: Optional[Callable] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        super().__init__(
            context_window=context_window,
            max_new_tokens=max_new_tokens,
            system_prompt=system_prompt,
            query_wrapper_prompt=query_wrapper_prompt,
            tokenizer_name=tokenizer_name,
            model_name=model_name,
            model=model,
            tokenizer=tokenizer,
            device_map=device_map,
            stopping_ids=stopping_ids,
            tokenizer_kwargs=tokenizer_kwargs,
            tokenizer_outputs_to_remove=tokenizer_outputs_to_remove,
            model_kwargs=model_kwargs,
            generate_kwargs=generate_kwargs,
            is_chat_model=is_chat_model,
            messages_to_prompt=messages_to_prompt,
            callback_manager=callback_manager,
        )
        model:AutoModelForCausalLM=self._model
        model.generation_config = GenerationConfig.from_pretrained(model_name)
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        model:AutoModelForCausalLM=self._model
        tokenizer:AutoTokenizer=self._tokenizer
        input_tensor=tokenizer.encode(prompt, return_tensors="pt")
        outputs = model.generate(input_tensor.to(model.device), max_new_tokens=self.max_new_tokens)
        response = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
        return CompletionResponse(text=response)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        model:AutoModelForCausalLM=self._model
        tokenizer:AutoTokenizer=self._tokenizer
        input_tensor=tokenizer.encode(prompt, return_tensors="pt")
        
        streamer = TextIteratorStreamer(tokenizer,skip_prompt=True)
        generation_kwargs = dict(inputs=input_tensor.to(model.device), streamer=streamer, max_new_tokens=self.max_new_tokens)
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        def gen() -> CompletionResponseGen:
            text = ""
            for x in streamer:
                text += x
                yield CompletionResponse(text=text, delta=x)

        return gen()

