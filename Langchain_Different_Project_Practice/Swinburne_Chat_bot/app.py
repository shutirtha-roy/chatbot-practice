from dotenv import load_dotenv

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain.document_loaders import PyPDFLoader

load_dotenv()

def get_vectordb(vector_path):
    embedder = OpenAIEmbeddings()
    vector_store = FAISS.load_local(vector_path, embedder, index_name="index", allow_dangerous_deserialization=True)
    return vector_store

def create_chain(vector_db):
    model = ChatOpenAI(model="gpt-4o" , temperature=0.5)

    prompt = ChatPromptTemplate.from_messages([
        ('system', """You are an AI assistant exclusively designed for Swinburne University students. Your sole purpose is to provide information directly related to Swinburne University. Here are your strict guidelines:

        1. ONLY answer questions based on the given context {context} about Swinburne University.
        2. If a question is not specifically about Swinburne University, DO NOT answer it. This includes but is not limited to:
           - General knowledge questions (e.g., "Who is the CEO of Google?")
           - Weather information
           - Current events not related to Swinburne
           - Any topic not directly concerning Swinburne University
        3. For non-Swinburne questions, respond with: "I'm sorry, but I can only provide information about Swinburne University. If you have any questions related to Swinburne, I'd be happy to help!"
        4. Be friendly and supportive, but maintain a professional tone when discussing Swinburne-related topics.
        5. If you're unsure about a Swinburne-related answer, say so and suggest where the student might find more information within the university.
        6. Do not provide personal opinions or advice. Stick strictly to official Swinburne University information.
        7. For sensitive Swinburne-related topics, direct students to appropriate university resources or support services.

        Remember, you are NOT a general-purpose AI. Your knowledge and responses are limited EXCLUSIVELY to Swinburne University-related information."""),
        MessagesPlaceholder('chat_history'),
        ('human', 'Gives greetings'),
        ('system', 'Hi! I am Swinburne Chat Bot. I am a chat assistant designed for the students of Swinburne University. How may I help you?'),
        ('human', '{input}')
    ])

    chain = create_stuff_documents_chain(
        prompt=prompt,
        llm=model
    )

    retriever = vector_db.as_retriever(
        search_type = "mmr", search_kwargs={'k': 1, "score_threshold": 0.1}
    )
    
    retrieval_chain = create_retrieval_chain(retriever, chain)

    return retrieval_chain

def process_chat(vector_db, chain, query, chat_history):
    response = chain.invoke({
        'input': query,
        'chat_history': chat_history
    })

    similarity_score = vector_db.similarity_search_with_score(query, search_type = "similarity", k=1)[0][1]

    # if similarity_score > 0.4:
    #     return "Sorry, I don't know the answer. If you have any specific questions or need information related to Swinburne University, feel free to ask!"

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