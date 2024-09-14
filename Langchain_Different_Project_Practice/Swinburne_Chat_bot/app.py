from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage

class VectorDB:
    def __init__(self, vector_path):
        self.vector_path = vector_path
        self.embedder = OpenAIEmbeddings()
        self.vector_store = None

    def load(self):
        self.vector_store = FAISS.load_local(
            self.vector_path, 
            self.embedder, 
            index_name="index", 
            allow_dangerous_deserialization=True
        )

    def as_retriever(self):
        return self.vector_store.as_retriever(
            search_type="similarity", 
            search_kwargs={'k': 1, "score_threshold": 0.1}
        )

    def similarity_search_with_score(self, query):
        return self.vector_store.similarity_search_with_score(query, search_type="similarity", k=1)

class ChatChain:
    def __init__(self, vector_db):
        self.vector_db = vector_db
        self.model = ChatOpenAI(model="gpt-4o", temperature=0.5)
        self.chain = None

    def create(self):
        prompt = ChatPromptTemplate.from_messages([
            ('system', 'You are a helpful assistant for Swinburne University students. Answer their questions according to the given context {context}.'),
            MessagesPlaceholder('chat_history'),
            ('human', 'Gives greetings'),
            ('system', 'Hi! I am Swinburne Chat Bot. I am a chat assistant designed for the students of Swinburne University. How may I help you?'),
            ('human', '{input}')
        ])

        stuff_chain = create_stuff_documents_chain(prompt=prompt, llm=self.model)
        retriever = self.vector_db.as_retriever()
        self.chain = create_retrieval_chain(retriever, stuff_chain)

    def invoke(self, query, chat_history):
        return self.chain.invoke({
            'input': query,
            'chat_history': chat_history
        })

class ChatProcessor:
    def __init__(self, vector_db, chat_chain):
        self.vector_db = vector_db
        self.chat_chain = chat_chain

    def process(self, query, chat_history):
        response = self.chat_chain.invoke(query, chat_history)
        similarity_score = self.vector_db.similarity_search_with_score(query)[0][1]

        if similarity_score > 0.45:
            return "Sorry, I don't know the answer. If you have any specific questions or need information related to Swinburne University, feel free to ask!"

        return response['answer']

class ChatHistory:
    def __init__(self):
        self.messages = []

    def add_message(self, message):
        self.messages.append(message)

    def get_messages(self):
        return self.messages

class ChatUI:
    def __init__(self):
        st.set_page_config(page_title="Chat With Swinburne FAQ", page_icon="🎓")
        st.header("Chat With Swinburne FAQ 🎓")

    def get_user_input(self):
        with st.form(key='user_input_form', clear_on_submit=True):
            user_input = st.text_input("Ask me anything:", placeholder="Type your message and press Enter")
            submit_button = st.form_submit_button(label='Send')
        return user_input if submit_button else None

    def display_chat_history(self, chat_history):
        for msg in reversed(chat_history.get_messages()):
            if isinstance(msg, HumanMessage):
                st.chat_message("user", avatar="🧑").markdown(f"**You:** {msg.content}")
            else:
                st.chat_message("assistant", avatar="🤖").markdown(f"**FAQ:** {msg.content}")

class ChatApplication:
    def __init__(self, vector_path):
        self.vector_db = VectorDB(vector_path)
        self.vector_db.load()
        self.chat_chain = ChatChain(self.vector_db)
        self.chat_chain.create()
        self.chat_processor = ChatProcessor(self.vector_db, self.chat_chain)
        self.chat_history = ChatHistory()
        self.chat_ui = ChatUI()

    def run(self):
        user_input = self.chat_ui.get_user_input()

        if user_input:
            with st.spinner('FAQ Chatbot is thinking...'):
                self.chat_history.add_message(HumanMessage(content=user_input))
                ai_output = self.chat_processor.process(user_input, self.chat_history.get_messages())
                self.chat_history.add_message(AIMessage(content=ai_output))

        self.chat_ui.display_chat_history(self.chat_history)

def main():
    load_dotenv()
    app = ChatApplication(vector_path='Swinburne_Chat_Bot')
    app.run()

if __name__ == '__main__':
    main()