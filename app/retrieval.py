from langchain_openai import ChatOpenAI # type: ignore

def create_qa_chain(vectorstore):
    llm = ChatOpenAI()

    def ask(query):
        docs = vectorstore.similarity_search(query, k=3)

        context = "\n".join([doc.page_content for doc in docs])

        prompt = f"""
        Answer the question based only on the context below:

        {context}

        Question: {query}
        """

        response = llm.invoke(prompt)
        return response.content

    return ask