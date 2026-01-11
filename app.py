import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

st.set_page_config(page_title="ServiceHive AI Assistant")

st.title("ServiceHive AI Assistant 🤖")

# Initialize LLM
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.3
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Ask something about ServiceHive")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get AI response
    response = llm.invoke(user_input)

    # Show AI message
    st.session_state.messages.append({"role": "assistant", "content": response.content})
    with st.chat_message("assistant"):
        st.markdown(response.content)
