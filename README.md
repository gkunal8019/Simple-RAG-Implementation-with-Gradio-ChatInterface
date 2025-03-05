# Simple RAG Implementation with Gradio ChatInterface

This repository demonstrates a simple Retrieval-Augmented Generation (RAG) implementation using the `llama_index` library integrated with a Gradio ChatInterface. The code is broken down into cells for clarity.

---

## Cell 1: Import Required Libraries

In this cell, we import all necessary libraries and modules required for document processing, indexing, retrieval, chat messaging, and creating the Gradio interface.

```python
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import StorageContext
from llama_index.core.storage.docstore.simple_docstore import SimpleDocumentStore
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.chat_engine.context import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.callbacks import CallbackManager
from llama_index.core import SimpleDirectoryReader
from llm_config import *
import gradio as gr
```

---

## Cell 2: Load Documents and Create the Index

This cell loads documents from the `html_markdown` directory, creates a vector store index from those documents, and initializes a storage context for later retrieval.

```python
# Load documents from the 'html_markdown' directory
dir_reader = SimpleDirectoryReader(input_dir="html_markdown")
documents = dir_reader.load_data()

# Create an index from the loaded documents
index = VectorStoreIndex.from_documents(documents)

# Initialize the storage context with the created index
storage_context = StorageContext.from_defaults(vector_store=index)
```

---

## Cell 3: Define the Query Processing Function

Here we define the `process_query` function. This function retrieves relevant document nodes based on the user query, merges their contents into a context string, and then sends this along with the query to the LLM for generating a response. The response is streamed back for a chat-like experience.

```python
def process_query(user_query, history):
    # Set up the base retriever with the desired number of similar documents to retrieve
    base_retriever = index.as_retriever(similarity_top_k=20)

    # Initialize the auto-merging retriever
    retriever = AutoMergingRetriever(base_retriever, storage_context, verbose=False)

    # Retrieve relevant nodes based on the user query
    nodes = retriever.retrieve(user_query)

    # Combine the content of the retrieved nodes into a single context string
    context_str = "\n\n".join([n.node.get_content() for n in nodes])

    # Create chat messages that include the context in the user prompt
    messages = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content="You are a helpful assistant."
        ),
        ChatMessage(
            role=MessageRole.USER,
            content=(
                "Using the following context, please answer the question below:\n"
                "---------------------\n"
                f"{context_str}\n"
                "---------------------\n"
                f"{user_query}"
            )
        ),
    ]

    # Send the chat messages to the LLM and collect the response
    response = nvllm.stream_chat(messages)
    full_response = ""
    try:
        for chunk in response:
            print(chunk.delta, end="", flush=True)  # Optionally print each chunk for real-time output
            full_response += chunk.delta
            yield full_response  # Yield partial response for ChatInterface streaming
    except Exception as e:
        yield f"Error occurred: {str(e)}"
    return full_response
```

---

## Cell 4: Create and Launch the Gradio Chat Interface

This final cell sets up the Gradio ChatInterface using the `process_query` function. It configures the interface’s title, description, and examples, and then launches it.

```python
iface = gr.ChatInterface(
    fn=process_query,
    title="AI Assistant",
    description="Ask me anything about your documents",
    examples=["list of departments in aiia", "Can you summarize this?"],
    type="messages"
)

iface.launch(share=True, debug=True)
```

---

## Dependencies

Ensure you have the following packages installed:
- `llama_index`
- `gradio`
- `nvllm` (or your preferred LLM integration)
- Additional dependencies as required by your `llm_config`

---

## Usage Instructions

1. **Clone the Repository:** Clone this repository to your local machine.
2. **Add Your Documents:** Place your documents in the `html_markdown` directory.
3. **Run the Cells:** Execute the code cells sequentially in a Jupyter notebook (or similar environment) to build the index and launch the chat interface.
4. **Interact:** Use the Gradio interface to ask questions and receive context-aware responses from your documents.

---
