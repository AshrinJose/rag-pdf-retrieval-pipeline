# Building an RAG Pipeline Using PDF Chunking and Retrieval

# Step 1: Install required packages
# pip install -U langchain langchain-openai langchain-community langchain-classic faiss-cpu tiktoken pypdf


# Step 2: Import Dependencies
import os
import time
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA


# Step 3: Load and chunk PDF documents
documents = []
# Folder containing PDFs
pdf_folder = "dataset"

# Load all PDFs
for file in os.listdir(pdf_folder):

    if file.endswith(".pdf"):

        pdf_path = os.path.join(pdf_folder, file)

        loader = PyPDFLoader(pdf_path)

        documents.extend(loader.load())

# split documents into chunks for embedding
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)


chunks = text_splitter.split_documents(documents)
print("Total chunks:", len(chunks))

chunk_lengths = [len(doc.page_content) for doc in chunks]
print("Average chunk size:",
      sum(chunk_lengths) / len(chunk_lengths))


# Step 4: Set Azure OpenAI credentials
os.environ["AZURE_OPENAI_API_KEY"] ="YOUR_AZURE_OPENAI_API_KEY"


# Step 5: Define embeddings
embeddings = AzureOpenAIEmbeddings(
 azure_endpoint="https://openai-api-management-gw.azure-api.net",
 api_version="2023-05-15",
 deployment="text-embedding-ada-002",
 api_key=os.environ["AZURE_OPENAI_API_KEY"]
)


# Step 6: Embed documents in batches to avoid rate limits and Build FAISS index manually from pre-computed embeddings
def embed_in_batches(chunks, embeddings, batch_size=5, delay=60):
    """Embed documents in batches to avoid rate limits."""
    all_texts = [chunk.page_content for chunk in chunks]
    all_metadatas = [chunk.metadata for chunk in chunks]

    all_embeddings = []

    for i in range(0, len(all_texts), batch_size):
        batch = all_texts[i:i + batch_size]
        print(f"Embedding batch {i//batch_size + 1}/{(len(all_texts) + batch_size - 1)//batch_size} "
              f"(chunks {i+1}-{min(i+batch_size, len(all_texts))})")

        try:
            batch_embeddings = embeddings.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"Rate limit hit, waiting {delay}s...")
            time.sleep(delay)
            batch_embeddings = embeddings.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)

        # Wait between batches (skip after last batch)
        if i + batch_size < len(all_texts):
            time.sleep(10)

    # Build FAISS index manually from pre-computed embeddings
    vectorstore = FAISS.from_embeddings(
        text_embeddings=list(zip(all_texts, all_embeddings)),
        embedding=embeddings,
        metadatas=all_metadatas
    )
    return vectorstore

vectorstore = embed_in_batches(chunks, embeddings, batch_size=5, delay=60)
print("Vectorstore created successfully!")


# Step 6: Initialize the Azure OpenAI LLM
llm = AzureChatOpenAI(
 azure_endpoint="https://openai-api-management-gw.azure-api.net",
 api_version="2025-01-01-preview",
 deployment_name="gpt-5-mini"
)


# Step 7: Create the RAG chain
qa_chain = RetrievalQA.from_chain_type(
 llm=llm,
 retriever=vectorstore.as_retriever(),
 return_source_documents=True
)


# Step 8: Ask a question
query = "what are the coverage limits mentioned in Involuntary Loss of Employment Policy? "
result = qa_chain.invoke({"query": query})


# Step 9: Print results
print("Answer:", result["result"])
print("\n--- Sources ---")
for i, doc in enumerate(result["source_documents"], 1):
 print(f"\nSource {i}:")
 print(doc.page_content)

