import streamlit as st
import autogen
import chromadb
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
import os
import asyncio

# Streamlit page configuration
st.set_page_config(page_title="AutoGen Group Chat with RAG", page_icon="🤖", layout="wide")

st.title("AutoGen Group Chat with RAG")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("OpenAI API Key", type="password")
    model = st.selectbox("Model", ["gpt-4", "gpt-3.5-turbo"], index=0)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agents_initialized" not in st.session_state:
    st.session_state.agents_initialized = False

# Function to initialize agents
def initialize_agents():
    config_list = [{"model": model, "api_key": api_key}]
    
    llm_config = {
        "config_list": config_list,
        "timeout": 60,
        "temperature": 0.8,
        "seed": 1234,
        "max_tokens": 2048
    }

    boss = autogen.UserProxyAgent(
        name="Boss",
        is_termination_msg=lambda x: isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper(),
        human_input_mode="NEVER",
        system_message="You are the user of operation of V93000. Your task is to initiate and oversee the process of generating a test method."
    )

    boss_aid = RetrieveUserProxyAgent(
        name="Boss_Assistant",
        is_termination_msg=lambda x: isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper(),
        human_input_mode="NEVER",
        retrieve_config={
            "task": "code",
            "docs_path": [
                "SmartRDI_overall.pdf"
            ],
            "chunk_token_size": 2000,
            "model": model,
            "collection_name": "groupchat",
            "get_or_create": True,
        },
        system_message="You are the Boss Assistant. Your primary task is to retrieve relevant information and documents from the specified knowledge base to support the generation of SmartRDI test method scripts."
    )

    coder = autogen.AssistantAgent(
        name="Test_Engineer",
        is_termination_msg=lambda x: isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper(),
        system_message="You are a Test Engineer specializing in semiconductor testing processes and highly skilled in C++. Your primary responsibility is to generate C++ code that accurately aligns with the SmartRDI syntax and structure.",
        llm_config=llm_config
    )

    reviewer = autogen.AssistantAgent(
        name="Code_Reviewer",
        is_termination_msg=lambda x: isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper(),
        system_message="You are a Code Reviewer specializing in reviewing code for semiconductor testing processes. Your primary responsibility is to review the generated C++ code to ensure it follows the SmartRDI syntax.",
        llm_config=llm_config
    )

    return boss, boss_aid, coder, reviewer

# Main chat interface
user_input = st.text_input("Enter your message:")

if user_input and api_key:
    if not st.session_state.agents_initialized:
        boss, boss_aid, coder, reviewer = initialize_agents()
        st.session_state.agents_initialized = True
        st.session_state.boss = boss
        st.session_state.boss_aid = boss_aid
        st.session_state.coder = coder
        st.session_state.reviewer = reviewer

    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Run AutoGen group chat
    groupchat = autogen.GroupChat(
        agents=[st.session_state.boss_aid, st.session_state.coder, st.session_state.reviewer],
        messages=[],
        max_round=20
    )
    manager = autogen.GroupChatManager(groupchat=groupchat)

    async def run_chat():
        await st.session_state.boss_aid.a_initiate_chat(
            manager,
            message=user_input,
            n_results=3,
        )

    # Run the chat
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_chat())
    loop.close()

    # Update chat display
    for message in groupchat.messages:
        st.session_state.messages.append({"role": message["name"], "content": message["content"]})

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar to start the chat.", icon="⚠️")