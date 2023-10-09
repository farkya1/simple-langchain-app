import streamlit as st
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


st.set_page_config(page_title="LangChain: Simple chatbot", page_icon="🦜")
st.title("Chatbot")

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")


llm_prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            # This prompt tells the chatbot how to respond. Try modifying it.
            "You are an impression bot. You will pick a random famous person and impersonate him. The user will try to get guess. If the user guesses right you will pick a new person to impersonate."
        ),
        HumanMessagePromptTemplate.from_template("{message}")
    ]
)

if prompt := st.chat_input(placeholder="Guess who I am"):
    st.chat_message("user").write(prompt)

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True)
    chain = LLMChain(
        llm=llm,
        prompt=llm_prompt,
        verbose=True
    )
    with st.chat_message("assistant"):
        response = chain({"message": prompt})
        st.write(response["text"])

