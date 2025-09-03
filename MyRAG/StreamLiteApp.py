import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Qwen3:4B • Ollama + LangChain", page_icon="🤖")

st.title("🤖 Qwen3:4B (Ollama) — Streamlit + LangChain")
st.caption("Minimal demo: PromptTemplate → ChatOllama → Text")

# Sidebar controls
with st.sidebar:
    st.header("Model Settings")
    model_name = st.text_input("Ollama model", value="qwen3:4b")
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
    top_p = st.slider("top_p", 0.1, 1.0, 0.95, 0.05)
    max_tokens = st.number_input("max tokens (0 = unlimited)", min_value=0, value=512, step=64)
    st.markdown("---")
    st.caption("Make sure `ollama pull qwen3:4b` was done.")

# Prompt template
system_prompt = (
    "You are a helpful assistant. Be concise and clear.\n"
    "If code is requested, provide minimal runnable examples."
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{user_input}")
])

# Model
llm = ChatOllama(
    model=model_name,
    temperature=temperature,
    top_p=top_p,
    num_predict=None if max_tokens == 0 else max_tokens,
)

# Chain (LCEL)
chain = prompt | llm | StrOutputParser()

# UI
user_text = st.text_area("Your message:", height=140, placeholder="Ask me anything…")
go = st.button("Generate")

if go and user_text.strip():
    with st.spinner("Thinking…"):
        try:
            # Stream token-by-token to the UI
            # LangChain's default run returns full string; we can iterate with .stream() for streaming.
            stream = chain.stream({"user_input": user_text})
            out = st.empty()
            acc = ""
            for chunk in stream:
                acc += chunk
                out.markdown(acc)
        except Exception as e:
            st.error(f"Error: {e}")
