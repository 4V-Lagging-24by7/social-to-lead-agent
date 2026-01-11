from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.fake import FakeEmbeddings



def load_vector_store():
    # Load product documents
    with open("data/product_docs.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # Split text into chunks
    splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = splitter.create_documents([text])

    # Lightweight embeddings (no torch / transformers)
    embeddings = FakeEmbeddings(size=384)

    # Create FAISS vector store
    vector_store = FAISS.from_documents(docs, embeddings)

    return vector_store
