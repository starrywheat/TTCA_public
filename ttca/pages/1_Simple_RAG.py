from __future__ import annotations

import streamlit as st
from langchain.callbacks import get_openai_callback
from Main import init_settings
from Main import init_vectordb
from Main import show_sidebar
from utils_llm import RAGpipeline_embfiter
from vectordb import PineconeVDB


show_sidebar()
init_settings()
tax_vdb, existing_indices = init_vectordb()

with st.form("choose_cases"):
    case_number = st.selectbox(
        "Choose existing case number :card_index_dividers:",
        tuple(st.session_state["existing_indices"]),
    )

    reviewer = st.text_input("Reviewer", "Your Name")

    # Load question
    query_template = """
    You are a tax expert. Based on the context provided, please determine if this case is about tax avoidance.
    Tax avoidance involves bending the rules of the tax system to try to gain a tax advantage that Parliament never intended. It often involves contrived, artificial transactions that serve little or no purpose other than to produce this advantage. It involves operating within the letter, but not the spirit, of the law.

    Please answer "Yes" if the case is about tax avoidance, "No" if the case is not about tax avoidance or "Don't know" if you don't know. Then provide a short summary of your reasoning. Use the following format in your response:

    Tax avoidance case: <Yes/No/Don't Know>
    Reasoning: <summary>
    """
    with st.expander("Expand to edit the prompt :memo::"):
        question_query = st.text_area("Question", query_template)
    submitted = st.form_submit_button("Submit")

if submitted:
    # Display the document
    # st_display_pdf(case_number)

    # Do the query
    vectorstore = tax_vdb.get_langchain_pinecone()

    # RAG pipeline
    ragpipeline = RAGpipeline_embfiter(
        st.session_state["chatmodel_name"],
        "sentence-transformers/all-mpnet-base-v2",
        vectorstore,
        topk=st.session_state["topk"],
        temperature=st.session_state["temperature"],
        similarity_threshold=st.session_state["similarity threshold"],
    )

    result, cb = ragpipeline.run_qachain(
        question_query,
        case_number,
        reviewer,
    )

    st.subheader("Answers :sunglasses: :")
    st.write(result["answer"])
    st.subheader("Sources :bookmark_tabs: :")
    with st.expander("Expand to see source paragraphs"):
        for d in result["source_documents"]:
            st.markdown(f"- {d.page_content}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total tokens:", cb.total_tokens)
    with col2:
        st.metric("Prompt tokens:", cb.prompt_tokens)
    with col3:
        st.metric("Completion tokens:", cb.completion_tokens)
    with col4:
        st.metric("Total costs :moneybag: :", f"$ {round(cb.total_cost,4)}")
