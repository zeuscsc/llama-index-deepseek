from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index import ServiceContext
from llama_index_extra_llm.deepseek import DeepSeekLLM
llm = DeepSeekLLM(
    model_name="deepseek-ai/deepseek-llm-7b-chat",
    tokenizer_name="deepseek-ai/deepseek-llm-7b-chat",
    context_window=3900,
    max_new_tokens=1024,
    generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    device_map="auto",
)
service_context = ServiceContext.from_defaults(llm=llm, embed_model="local:BAAI/bge-small-en-v1.5")
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents, service_context=service_context)
query_engine = index.as_query_engine(streaming=True, similarity_top_k=1)
prompt=DeepSeekLLM.messages2prompt([])
streaming_response=query_engine.query(prompt)
streaming_response.print_response_stream()