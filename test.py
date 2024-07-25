import os
import asyncio
import streamlit as st
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent

# Load configuration from JSON
config_list = autogen.config_list_from_json(
    env_or_file="/Users/apple/Desktop/GenAI/TE_asis/pre/OAI_CONFIG_LIST.json",
    filter_dict={"model": ["gpt-4o"]}
)

def termination_msg(x):
    return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()

llm_config = {"config_list": config_list, "timeout": 60, "temperature": 0.8, "seed": 1234,"max_tokens": 2048}

# Define agents with their respective configurations and system messages
boss = UserProxyAgent(
    name="Boss",
    is_termination_msg=termination_msg,
    human_input_mode="NEVER",
    code_execution_config=False,
    default_auto_reply="Reply `TERMINATE` if the task is done.",
    system_message="""
    You are the user of operation of V93000. 
    Your task is to initiate and oversee the process of generating a test method. 
    Coordinate with all agents, ensuring that each one completes their tasks effectively.
    Ensure that the final output meets the requirements.
    Reply `TERMINATE` when the task is done.""",
    description="The boss who asks questions and assigns tasks."
)

boss_aid = RetrieveUserProxyAgent(
    name="Boss_Assistant",
    is_termination_msg=termination_msg,
    human_input_mode="NEVER",
    default_auto_reply="Reply `TERMINATE` if the task is done.",
    max_consecutive_auto_reply=3,
    retrieve_config={
        "task": "code",
        "docs_path": [
            "SmartRDI_overall.pdf"
        ],
        "chunk_token_size": 2000,
        "model": config_list[0]["model"],
        "collection_name": "groupchat",
        "get_or_create": True,
    },
    code_execution_config=False,
    system_message="""
    You are the Boss Assistant. 
    Your primary task is to retrieve relevant information and documents from the specified knowledge base to support the generation of SmartRDI test method scripts.
    Follow these steps:
    1. Identify relevant sections within the document that describe the general SmartRDI code structure and specific test method scripts.
    2. Extract code blocks related to the test methods (e.g., Leakage Test, Functional Test, DC Test). Ensure the extracted code follows the SmartRDI syntax and structure.
    3. If no relevant document is found, use your general knowledge base to combine existing data with the SmartRDI structure to support the generation of the requested code.
    
    Reply `TERMINATE` when the task is done, indicating that the required information has been successfully retrieved and processed.
    """,
    description="Assistant with the capability to retrieve and process additional content to solve complex problems related to SmartRDI test method code generation."
)

coder = AssistantAgent(
    name="Test_Engineer",
    is_termination_msg=termination_msg,
    system_message="""
    You are a Test Engineer specializing in semiconductor testing processes and highly skilled in C++.
    Your primary responsibility is to generate C++ code that accurately aligns with the SmartRDI syntax and structure. Follow these detailed instructions:
    
    1. General Code Structure: Ensure the generated code follows the SmartRDI: Code structure provided by Boss_Asistant. Use it as a framework and template.
    2. Insert Specific Test Method Code Block: Replace the specific code for different test method (e.g., Leakage Test, Functional Test, DC Test) into the designated section within the SmartRDI structure. Ensure the code inside run() and SMC_backgroundProcessing() corresponds to different test method.
    3. Customization Based on User Input: When user define value for variables, allow the customization. Ensure these values are used in the code generation.
    5. Integrate the retrieved context and code blocks provided by Boss Assistant to ensure accuracy.
    
    
    Reply `TERMINATE` in the end when everything is done.""",
    llm_config={
        "timeout": 600,
        "cache_seed": 42,
        "config_list": config_list,
    },
    description="Senior C++ Test Engineer who generates code following the SmartRDI syntax, ensuring accuracy, optimization, and customization based on user input for semiconductor verification testing processes."
)

tem = AssistantAgent(
    name="Test_Engineering_Manager",
    is_termination_msg=termination_msg,
    system_message="""
    You are a Test Engineering Manager specializing in semiconductor testing processes. 
    Your primary responsibility is to design, plan, and oversee the execution of the test engineering process, ensuring that it aligns with the project goals and meets all specified requirements. 
    Coordinate efforts with the Senior Test Engineer and other team members to ensure quality and manage risks. 
    Your tasks include requirement gathering, planning, design review, quality assurance, documentation, and reporting.
    Ensure all deliverables are met on time and adhere to industry standards.
    Reply `TERMINATE` when the task is done.""",
    llm_config=llm_config,
    description="Product Manager who can design and plan the project.",
)

reviewer = AssistantAgent(
    name="Code_Reviewer",
    is_termination_msg=termination_msg,
    system_message="""
    You are a Code Reviewer specializing in reviewing code for semiconductor testing processes. 
    Your primary responsibility is to review the generated C++ code to ensure it follows the SmartRDI syntax. 
    Ensure the final code is accurate, and meets the project requirements.
    Reply `TERMINATE` when the task is done.""",
    llm_config=llm_config,
    description="Code Reviewer who can review the code following the SmartRDI syntax and strucure. Meanwhile, you can also provide feedback and suggestions for specific tasks."
)

def _reset_agents():
    boss.reset()
    boss_aid.reset()
    coder.reset()
    tem.reset()
    reviewer.reset()

# Streamlit app setup
st.title("SmartRDI Code Generation")

st.write("Interact with the SmartRDI coding assistant to generate test method scripts.")

problem = st.text_area("Enter the problem statement:", value="Generate the leakage test method code using SmartRDI syntax. Now i want to set VDD33 = 5, VSS = 0.5; iRange to 100mA. Also I don't want to include any settling time")

if st.button("Generate Code"):
    _reset_agents()
    groupchat = GroupChat(
        agents=[boss_aid, tem, coder, reviewer], messages=[], max_round=20, speaker_selection_method="round_robin"
    )
    manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)
    boss_aid.initiate_chat(
        manager,
        message=f"Retrieve relevant information and documents to support the following task: {problem}",
        problem=problem,
        n_results=3,
    )
    for msg in groupchat.messages:
        st.write(f"{msg['name']}: {msg['content']}")