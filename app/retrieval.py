def create_qa_chain(vectorstore):

    def ask(query):
        docs = vectorstore.similarity_search(query,k=3)

        context = "\n\n".join([doc.page_content for doc in docs])

        return {
            "query": query,
            "retrieved_context": context
        }
    return ask