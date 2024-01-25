from __future__ import annotations

import streamlit as st
from langchain.callbacks import get_openai_callback
from langchain.callbacks import StreamlitCallbackHandler
from Main import init_settings
from Main import show_sidebar

from utils_llm import agent_rag_pipeline


def init_qa_agent(
    reset: bool = False,
    reviewer: str = "",
    case_number: str = "",
):
    # Setup the chat
    if "messages" not in st.session_state.keys() or reset:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": f"Hello {reviewer}! Ask me anything about the tax tribunal case {case_number}.",
            },
        ]
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Setup the llm
    vectorstore = st.session_state["tax_vdb"].get_langchain_pinecone()
    if "qa_agent" not in st.session_state or reset:
        st.session_state["qa_agent"] = agent_rag_pipeline(
            st.session_state["chatmodel_name"],
            vectorstore,
            case_number,
            topk=st.session_state["topk"],
            temperature=st.session_state["temperature"],
        )


show_sidebar()
init_settings()


with st.form("choose_cases"):
    case_number = st.selectbox(
        "Choose existing case number :card_index_dividers:",
        tuple(st.session_state["existing_indices"]),
    )
    
    reviewer = st.text_input("Reviewer", "Your Name")
    submitted = st.form_submit_button("Submit")


if submitted:
    init_qa_agent(reset=True, reviewer=reviewer, case_number=case_number)

if prompt := st.chat_input(placeholder="Is this a tax avoidance case?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        with get_openai_callback() as cb:
            st_cb = StreamlitCallbackHandler(
                st.container(),
                expand_new_thoughts=False,
            )
            result = st.session_state["qa_agent"](
                {"input": prompt},
                tags=[reviewer],
                metadata={"case_number": case_number},
                callbacks=[st_cb],
            )
            st.session_state.messages.append(
                {"role": "assistant", "content": result["output"]},
            )
            st.write(result["output"])
            st.metric(
                "Total costs :moneybag: :",
                f"$ {round(cb.total_cost,4)}",
            )
            # # Add the new run to langsmith dataset
            # add_to_dataset()

clear_chat = st.button(
    "Clear chat",
    on_click=init_qa_agent,
    kwargs={"reset": True},
)
