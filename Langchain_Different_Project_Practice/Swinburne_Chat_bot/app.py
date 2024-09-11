from dotenv import load_dotenv

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage


load_dotenv()

def get_vectordb(vector_path):
    embedder = OpenAIEmbeddings()
    vector_store = FAISS.load_local(vector_path, embedder, index_name="index", allow_dangerous_deserialization=True)
    return vector_store


def create_chain(vector_db):
    model = ChatOpenAI(model="gpt-4o" , temperature=0.5) # set the model you are wanting to use

    prompt = ChatPromptTemplate.from_messages([
        ('system', 'You are a helpful assistant for Swinburne University students. Answer their questions according to the given context {context}.'),
        MessagesPlaceholder('chat_history'),
        ('human', 'Gives greetings'),
        ('system', 'Hi! I am BOKU. I am a chat assistant designed for the students of Swinburne University. How may I help you?'),
        ('human', '{input}')
    ])

    chain = create_stuff_documents_chain(
        prompt=prompt,
        llm=model
    )

    retriever = vector_db.as_retriever(
        search_type = "mmr", search_kwargs={'k': 1, "score_threshold": 0.1}
    ) #set the parameter according to your need
    
    #print(vector_db.similarity_search_with_score("Swinburne", search_type = "mmr", k=1))
    
    retrieval_chain = create_retrieval_chain(retriever, chain)
    #print([doc for doc in retrieval_chain])
    #print(retriever)

    return retrieval_chain

def process_chat(vector_db, chain, query, chat_history):
    response = chain.invoke({
        'input': query,
        'chat_history': chat_history
    })

    similarity_score = vector_db.similarity_search_with_score(query, search_type = "mmr", k=1)[0][1]

    if similarity_score > 0.45:
        return "Sorry, I don't know the answer. If you have any specific questions or need information related to Swinburne University, feel free to ask!"

    return response['answer']


vector_db = get_vectordb(vector_path = 'Swinburne_Chat_Bot')
chain = create_chain(vector_db)

def main():
    st.set_page_config(page_title="Chat With Swinburne FAQ", page_icon="🎓")
    st.header("Chat With Swinburne FAQ 🎓")

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    with st.form(key='user_input_form', clear_on_submit=True):
        user_input = st.text_input("Ask me anything:", placeholder="Type your message and press Enter")
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        with st.spinner('FAQ Chatbot is thinking...'):
            st.session_state['chat_history'].append(HumanMessage(content=user_input))
            ai_output = process_chat(vector_db, chain, user_input, st.session_state['chat_history'])

            st.session_state['chat_history'].append(AIMessage(content=ai_output))

    for msg in reversed(st.session_state['chat_history']):
        if isinstance(msg, HumanMessage):
            st.chat_message("user", avatar="🧑").markdown(f"**You:** {msg.content}")
        else:
            st.chat_message("assistant", avatar="🤖").markdown(f"**FAQ:** {msg.content}")

if __name__ == '__main__':
    main()
