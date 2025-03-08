from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.llms import ChatMessage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.vllm import Vllm
from llama_index.core import StorageContext,load_index_from_storage
import os

#初始化一个HuggingFaceEmbedding对象，将文本向量化
embed_model = HuggingFaceEmbedding(
    model_name="/teacher_data/zhangyang/llm/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/Ceceliachenen/paraphrase-multilingual-MiniLM-L12-v2"
)

Settings.embed_model = embed_model

llm = HuggingFaceLLM(
        model_name="/teacher_data/zhangyang/llm/Qwen/Qwen1___5-1___8B-Chat",
        tokenizer_name="/teacher_data/zhangyang/llm/Qwen/Qwen1___5-1___8B-Chat",
        model_kwargs={"trust_remote_code": True},
        tokenizer_kwargs={"trust_remote_code": True}
    )

Settings.llm = llm

#判断本地数据库是否存在
if not os.path.exists("storage"):
    documents = SimpleDirectoryReader("/root/app/project/demo_20241219/data").load_data()
    node_parser = SimpleNodeParser.from_defaults(chunk_size=1024)
    base_node = node_parser.get_nodes_from_documents(documents=documents)
    index = VectorStoreIndex.from_documents(documents)
    #存储索引
    index.storage_context.persist()
    print("未发现本地数据集，开始创建新的数据库并存index！")
else:
    print("加载已存在的index数据库！！")
    #加载现有索引
    storage_context = StorageContext.from_defaults(persist_dir="storage")
    #从存储中加载索引
    index = load_index_from_storage(storage_context = storage_context)
query_engine = index.as_query_engine()

rsp = query_engine.query("OpenCompass是什么？")

# rsp = llm.chat(messages=[ChatMessage(content="传感器在中国有哪些品牌")])
print(rsp)
# #指定目录读取文档
# documents = SimpleDirectoryReader("/root/app/project/demo_20241219/data").load_data()
# # print("===========================")
# # print(documents)
# # print("======================")
# #创建节点解析器
# node_parser = SimpleNodeParser.from_defaults(chunk_size=1024)

# #将文档分割成节点
# base_node = node_parser.get_nodes_from_documents(documents=documents)
# print("================")
# print(base_node)
# print("================")




# # index = VectorStoreIndex.from_documents(documents)
# index = VectorStoreIndex(nodes=base_node)

# query_engine = index.as_query_engine()

# rsp = query_engine.query("OpenCompass是什么？")

# # rsp = llm.chat(messages=[ChatMessage(content="传感器在中国有哪些品牌")])
# print(rsp)

