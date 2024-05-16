from rich import print
import os
import sys
import logging

from llama_index.core import (Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

# By default, ada v2 is used for embedding.
# The new text-embedding-3-small is more capable, and 1/5th the price.
Settings.llm = OpenAI(model="gpt-4o")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.text_splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)
PERSIST_DIR = "./storage"

STATES = [
    "Arizona",
    "Colorado",
    "Delaware",
    "Florida",
    "Georgia",
    "Illinois",
    "Indiana",
    "Iowa",
    "Kentucky",
    "Maryland",
    "Michigan",
    "Minnesota",
    "New Mexico",
    "North Carolina",
    "Ohio",
    "Pennsylvania",
    "South Carolina",
    "Tennessee",
    "Virginia",
    "Washington DC",
    "West Virginia",
    "Wisconsin",
]


def load():
    """Stage 1: Load Data"""
    documents = SimpleDirectoryReader("data").load_data()
    return documents

def index(documents):
    """Stage 2: Index Data"""
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    index.storage_context.persist(persist_dir=PERSIST_DIR)


if __name__ == "__main__":

    # See if the index has been created, and create it if it hasn't
    if not os.path.exists(PERSIST_DIR):
        index(load())
    
    prompts = [
        "How does Stalin's 'niet' ability work?",
        "Teach me about the Space Race mechanic.",
        "Which civil war battles are covered in the game?",
        "How many conferences are played in the Training Scenario?",
        "What are the victory conditions for the 1862 scenario?"
    ]

    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine()
    for prompt in prompts:
        response = query_engine.query(f"{prompt}")
        print(f"\nPrompt: {prompt}")
        print(f"\nAnswer: {response.response}\n")
        for node in response.source_nodes:
            filename = node.node.metadata["file_name"]
            page = node.node.metadata["page_label"]
            print(f"Citation: {filename} - page {page}")