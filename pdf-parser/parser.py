import os
from rich import print
import logging
import sys
import os.path
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

os.environ["OPENAI_API_KEY"] = "****"

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# check if storage already exists
PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine()
response = query_engine.query("How many conferences are played in a tournament scenario?")

print("---SHORT RESPONSE---")
print(response.response)

for node in response.source_nodes:
    print("-----")
    text_fmt = node.node.get_content().strip().replace("\n", " ")[:1000]
    print(f"Text:\t {text_fmt} ...")
    print(f"Metadata:\t {node.node.metadata}")
    print(f"Score:\t {node.score:.3f}")