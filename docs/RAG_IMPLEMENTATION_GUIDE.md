# RAG Implementation — Technical Guide

This document explains how Retrieval-Augmented Generation (RAG) is implemented in the customer support agent.

## Overview

The LLM agent does not rely solely on its training data to answer questions. Instead, before generating a response, it retrieves the most relevant documents from a local knowledge base and injects them as context. This keeps answers accurate and grounded in the actual support policy.

## Knowledge Base

**16 documents** across 8 categories, stored in ChromaDB:

| Category | Documents |
|---|---|
| `returns` | Return policy, return process steps, non-returnable items |
| `shipping` | Shipping methods & times, international shipping, order tracking |
| `support` | Contact information, response times |
| `warranty` | Product warranty |
| `technical` | Technical support |
| `account` | Account management |
| `orders` | Order modifications |
| `payment` | Payment methods |
| `billing` | Billing and invoices |
| `products` | Product availability, size guide |

## Embedding & Storage

Documents are embedded once at startup using `sentence-transformers/all-MiniLM-L6-v2` and stored in a persistent ChromaDB collection at `data/chroma_db/`. Subsequent server restarts skip re-ingestion if the collection already has data.

```python
self.chroma_client = chromadb.PersistentClient(path="./data/chroma_db")
self.collection = self.chroma_client.get_or_create_collection("customer_support_kb")
self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
```

## Query Flow

At query time, ChromaDB's built-in embedding pipeline handles the query embedding and cosine similarity search:

```python
results = self.collection.query(
    query_texts=[query],
    n_results=3,
    include=['documents', 'metadatas', 'distances']
)
```

The top-3 results are formatted and injected as tool output into the ReAct agent's scratchpad.

## ReAct Agent Integration

The RAG search is exposed as a LangChain `Tool` named `knowledge_search`. The agent uses the ReAct (Reason + Act) loop to decide when to call it:

```
Question: What is the return policy?
Thought: I need to look up the return policy information.
Action: knowledge_search
Action Input: return policy
Observation: **Return Policy Overview** (Category: returns, Relevance: 94.2%)
              We offer a 30-day return policy...
Thought: I now know the final answer.
Final Answer: Our return policy allows...
```

The agent can call the tool multiple times per query if it needs more information.

## Result Formatting

Each retrieved document is formatted with title, category, and relevance score:

```python
for doc, meta, distance in zip(documents, metadatas, distances):
    relevance = round((1 - distance) * 100, 1)
    formatted_results.append(
        f"**{meta['title']}** (Category: {meta['category']}, Relevance: {relevance}%)\n{doc}"
    )
```

## Testing the Knowledge Base

```bash
python src/utils/kb_test.py
```

This prints the collection structure, runs sample queries, and shows what the RAG retrieval returns for each.

## Sample Queries

| Query | Expected Category |
|---|---|
| "What is your return policy?" | `returns` |
| "How long does shipping take?" | `shipping` |
| "How can I contact support?" | `support` |
| "What payment methods do you accept?" | `payment` |
| "Do you ship internationally?" | `shipping` |
| "What warranty do you offer?" | `warranty` |
| "How do I track my order?" | `shipping` |
| "Can I cancel my order?" | `orders` |
