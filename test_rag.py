from utils.rag import load_vector_store

vs = load_vector_store()

results = vs.similarity_search("What pricing plans are available?")

for r in results:
    print(r.page_content)
