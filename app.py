import os
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv


DB_DIR = "./chroma_db_cosine"
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"batch_size": 32, "normalize_embeddings": True},
)

load_dotenv() #load GOOGLE_API_KEY from .env file

#llm
llm = ChatGoogleGenerativeAI( model = "gemini-3.5-flash-lite", temperature =0)

vectorstore = Chroma( persist_directory = DB_DIR, embedding_function= embeddings,collection_metadata={"hnsw:space": "cosine"} )
print("✅ Total chunks loaded:", vectorstore._collection.count())

# retriever = vectorstore.as_retriever(search_kwargs= {"k":5})
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.4,  # Minimum similarity score (0.0 to 1.0)
        "k": 5                   # Optional: max number of documents to return
    }
)

prompt = ChatPromptTemplate.from_template(
    """ Answer the question based ONLY on the following context
    
    Rules:
    - Cite the relevant page number(s) inline as [Page X] when stating facts.
    - Only cite pages that actually contributed to your answer.
    - At the very end of your response, add a single line: 
      "📄 Sources: Referenced from page(s) X, Y."

    Context: {context}

    Question: {question}
    
    """)

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)
def format_docs(docs):
    formatted = []
    for doc in docs:
        # Remove '+ 1' if your Chroma metadata already uses 1-indexed pages
        page = doc.metadata.get("page", None)
        page_label = f"Page {page + 1}" if page is not None else "Unknown Page"
        formatted.append(f"[{page_label}]\n{doc.page_content}")
    return "\n\n".join(formatted)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def main():
    # rag_chain, retriever = build_rag_chain()
    print("\nHands on ML RAG CLI ready! Type your question below (or 'exit' to quit).\n")

    while True:
        try:
            query = input("Query > ").strip()
            if not query:
                continue
            if query.lower() in {"exit", "quit", "q"}:
                print("Goodbye!")
                break

            print("\nAnswer:\n", rag_chain.invoke(query))
            

        except KeyboardInterrupt:
            print("\nThanks!")
            break


if __name__ == "__main__":
    main()
