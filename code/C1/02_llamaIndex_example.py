import os
# os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings 
from llama_index.llms.ollama import Ollama
# from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


load_dotenv()

# 使用 AIHubmix
# Settings.llm = OpenAILike(
#     model="glm-4.7-flash-free",
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     api_base="https://aihubmix.com/v1",
#     is_chat_model=True
# )

# Settings.llm = OpenAI(
#     model="deepseek-chat",
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     api_base="https://api.deepseek.com"
# )

Settings.llm = Ollama(
    model="deepseek-r1:8b",
    base_url="http://127.0.0.1:11434",
    request_timeout=360.0,
    additional_kwargs={"num_ctx": 4096}
)

Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

_base_dir = os.path.dirname(__file__)
_md_path = os.path.abspath(os.path.join(_base_dir, "../../data/C1/markdown/easy-rl-chapter1.md"))
docs = SimpleDirectoryReader(input_files=[_md_path]).load_data()

index = VectorStoreIndex.from_documents(docs)

query_engine = index.as_query_engine()

print(query_engine.get_prompts())

print(query_engine.query("文中举了哪些例子?"))
