# learn-llama-index
RAG has been an interesting journey. I'm in the process of writing a RAG application, and I'm learning a lot. The [High-Level Concepts](https://docs.llamaindex.ai/en/stable/getting_started/concepts/?h=high+level+concepts) guide from the LlamaIndex docs is a great place to start. They discuss, in general, the five 'stages' every RAG app will have:
 - Loading
 - Indexing
 - Storing
 - Querying
 - Evaluating
What you quickly discover, is each of these stages may have several steps (pipelines), and configuration varies widely.  

In the case of Querying, there are generally three sub-stages: Retrieval, Post Processing, Response Synthesis. If we look at Retrieval, LlamaIndex covers 17 different [Retrievers](https://docs.llamaindex.ai/en/stable/api_reference/retrievers/) in their docs alone. Any of these can be used singularly, or combined for various effects and tactics you may want to employ when RAG-ing your data.

## Loading Data
Loading your data is part of the "Ingestion Pipeline" which includes loading -> transforming -> indexing and storage. I've been scoping my loading stage to _just_ use LlamaIndex's SimpleDirectoryReader because it's straightforward to setup, and is general enough for my use case. Here's how we can load some data:

```python
from llama_index.core import (SimpleDirectoryReader)


def load():
    """Stage 1: Load Data"""
    documents = SimpleDirectoryReader("data").load_data()
    return documents

if __name__ == "__main__":
    print(load())
```

My data consists of rule books from GMT war games. Specifically; Twilight Struggle, Churchill, and For the People. The data returned from `SimpleDirectoryReader` is a list of "Documents". The LlamaIndex definition of a Document is as follows:

> A Document is a generic container around any data source - for instance, a PDF, an API output, or retrieved data from a database. They can be constructed manually, or created automatically via our data loaders.

In our case, the load stage returned a list of 116 documents which happens to equal the total pages for all three rule books. This must be how LlamaIndex handles PDFs when using the `SimpleDirectoryReader`. The content of a document looks like this:

```Python
Document(
        id_='c7adc363-32e7-42b3-99fe-b356cd2f0be7',
        embedding=None,
        metadata={'page_label': '22', 'file_name': 'CHURCHILLRules-Final.pdf', 'file_path': 'C:\\projects\\learn-llama-index\\advanced-rag\\data\\CHURCHILLRules-Final.pdf', 'file_type': 'application/pdf', 'file_size': 2051633, 'creation_date': '2024-05-12', 'last_modified_date': '2024-05-12'},
        excluded_embed_metadata_keys=['file_name', 'file_type', 'file_size', 'creation_date', 'last_modified_date', 'last_accessed_date'],
        excluded_llm_metadata_keys=['file_name', 'file_type', 'file_size', 'creation_date', 'last_modified_date', 'last_accessed_date'],
        relationships={},
        text='Churchill Rules22\n© 2015 GMT Games, LLC\n10.0 Secret Agenda Variant \nDESIGN NOTE: I have found gamers of two minds on games \nwith continuous scoring like Churchill. Some folks like know-ing the exact score at all times and manage their decisions based on perfect ‘score’ information, whileothers like a bit more uncertainty. The core game uses perfect scoring but for those who like to have a bit more bluff in their games I offer the following official variant.\n10.1 Secret Agenda Markers\n ‘Secret Agenda’ markers are marked as such and also have the name of a country or colony on one side. Note that there are some duplicates for some countries/colonies; this is intentional and these are not extras.\n10.2 Secret Agenda Procedure\nTake the 36 Secret Agenda markers; each player secretly and randomly draws three markers. Do not show them to your op-ponents. At the end of the game before you determine the winner, all players reveal their Secret Agenda markers.10.3 Secret Agenda Scoring\nIf at the end of the game a player has a Political Alignment marker in a country/colony that matches one of their Secret Agenda markers they score five additional points per marker, for a potential of 15 points if all three markers meet this condition. After these points have been applied to each players score the winner is determined.\nPLAY NOTE: It is possible and intentional that the application of the Secret Agenda marker bonus could impact the condition under which the winner is determined, i.e.,    creates a 15 point difference in score that can change the winner. This fact needs to be incorporated into a player’ s strategy, so be careful how hard you fight for your Secret Agenda.\nPLAY NOTE: You score 5 VP per Secret Agenda marker, so if you have two markers for the same location, you would score 10 VP .\nStalin and Churchill enjoy a private moment during the Yalta Conference.',
        start_char_idx=None,
        end_char_idx=None,
        text_template='{metadata_str}\n\n{content}',
        metadata_template='{key}: {value}',
        metadata_seperator='\n'
```

The metadata returned here is pretty handy. My hope is that I'll be able to use this to accurately cite answers later.

## Indexing and Storage
I switched the embedding model from ada v2 to the new `text-embedding-3-small` because it's [more capable](https://platform.openai.com/docs/guides/embeddings/embedding-models), and 1/5th of the price. Changing the default chunk size, and overlap has a significant change on the embeddings. OpenAI is using the `text-embedding-3-large` model, with chunk sizes of 800 and 50% overlap per this: [How it Works](https://platform.openai.com/docs/assistants/tools/file-search/how-it-works). While `text-embedding-3-large` is a more capable model, it takes up significantly more disk.

#### text-embedding-3-small
```
Default settings:                   5 files - 7.75mb
chunk_size=1000, chunk_overlap=200: 5 files - 9.55mb
chunk_size=800, chunk_overlap=400:  5 files - 11.5mb
chunk_size=200, chunk_overlap=100:  5 files - 64.9mb
```

#### text-embedding-3-large
```
Default settings:                   5 files - 14.3mb
chunk_size=1000, chunk_overlap=200: 5 files - 17.9mb
chunk_size=800, chunk_overlap=400:  5 files - 21.5mb
chunk_size=200, chunk_overlap=100:  5 files - 124mb
```

#### text-embedding-ada-002
```
Default settings:                   5 files - 7.78mb
chunk_size=1000, chunk_overlap=200: 5 files - 9.61mb
chunk_size=800, chunk_overlap=400:  5 files - 11.6mb
chunk_size=200, chunk_overlap=100:  5 files - 65.3mb
```

It's interesting that the embeddings are all larger than the actual files (`7.04mb`). Surely corporations will need to consider this when performing RAG on private data.


