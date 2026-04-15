import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI 
from ragas.testset import TestsetGenerator  
from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings as RagasHuggingFaceEmbeddings

api_key = "u0rgjezq53e5tv01000dg95v0yqn2eecv02b7z3r"
base_url = "https://api.gpugeek.com/v1"

llm = ChatOpenAI(
    api_key=api_key,
    base_url=base_url,
    model="DeepSeek/DeepSeek-V3.1-0821",
    temperature=0.7
)
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device': 'cuda'},
    encode_kwargs={
        'normalize_embeddings': True,
        'batch_size': 256
    }
)

generator = TestsetGenerator(
    llm=llm,  # 直接传 LangChain LLM
    embedding_model=embeddings  # 直接传 LangChain Embeddings
)

loader = DirectoryLoader("/data/wkJIN/PHY_Master/LANDAU/prior/out/Advanced Mathematical Methods_ Conformal Field Theory", glob="**/*.md")
documents = loader.load()

testset = generator.generate_with_langchain_docs(
    documents=documents,
    testset_size=5,  # 参数名是 testset_size 不是 test_size
    with_debugging_logs=True
)

df = testset.to_pandas()
print(df.head())
df.to_csv("rag_evaluation_testset.csv", index=False)