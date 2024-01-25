"""
Use this streamlit app for prompt experimentation with 2 subpages

1) simple RAG implementation
2) Conversational RAG agent

The main page allow for uploading documents to Azure Blob add to a vectorDB index

Author:
Daisy Mak <starrywheat.dm@gmail.com>

Updated on:
21 Jan 2024
"""
from __future__ import annotations

import base64

import streamlit as st
from prep_doc import download_blob
from prep_doc import text_to_vectordb
from prep_doc import upload_blob_data
from vectordb import PineconeVDB


def st_display_pdf(case_number: str):
    filebyte = download_blob(
        st.session_state["blob_container_name"],
        f"{case_number}.pdf",
    )

    base64_file = base64.b64encode(filebyte).decode("utf-8")

    file_display = (
        f'<embed src="data:application/pdf;base64,{base64_file}" '
        'width="800" height="500" type="application/pdf"></embed>'
    )
    st.markdown(file_display, unsafe_allow_html=True)


@st.cache_resource
def init_settings():
    # Settings
    if "chatmodel_name" not in st.session_state:
        st.session_state["chatmodel_name"] = "gpt-3.5-turbo"
    if "index_name" not in st.session_state:
        st.session_state["index_name"] = "tax-vdb"
    if "blob_container_name" not in st.session_state:
        st.session_state["blob_container_name"] = "casedocuments"
    if "embed_model_name" not in st.session_state:
        st.session_state["embed_model_name"] = "BAAI/bge-base-en"

    # load vectordb
    tax_vdb = PineconeVDB(
        st.session_state["index_name"],
        st.session_state["embed_model_name"],
    )
    existing_indices = tax_vdb.get_namespaces()
    if "existing_indices" not in st.session_state:
        st.session_state["existing_indices"] = existing_indices
    if "tax_vdb" not in st.session_state:
        st.session_state["tax_vdb"] = tax_vdb

    # Developer parameters
    if "topk" not in st.session_state:
        st.session_state["topk"] = 5
    if "temperature" not in st.session_state:
        st.session_state["temperature"] = 0.0


@st.cache_resource
def init_vectordb():
    # load vectordb
    tax_vdb = PineconeVDB(
        st.session_state["index_name"],
        st.session_state["embed_model_name"],
    )
    existing_indices = tax_vdb.get_namespaces()

    # Developer parameters
    if "topk" not in st.session_state:
        st.session_state["topk"] = 5
    if "temperature" not in st.session_state:
        st.session_state["temperature"] = 0.2
    if "similarity threshold" not in st.session_state:
        st.session_state["similarity threshold"] = 0.5
    return tax_vdb, existing_indices


def show_sidebar():
    with st.sidebar:
        st.subheader("Developer corner")
        st.session_state["topk"] = st.slider(
            "Number of relevant text to retrieved",
            1,
            10,
            5,
        )
        st.session_state["temperature"] = st.slider(
            "Temperature of LLM",
            0.0,
            1.0,
            0.2,
        )
        st.session_state["similarity threshold"] = st.slider(
            "Similarity Threshold for retrieval",
            0.0,
            1.0,
            0.5,
        )


init_settings()


with st.form("Upload_file"):
    upload_file = st.file_uploader(
        "Upload a file :card_index_dividers:",
    )
    submitted = st.form_submit_button("Click to upload to VectorDB")
    if upload_file and submitted:
        with st.spinner("Loading the upload documents into VectorDB"):
            filename = upload_blob_data(
                st.session_state["blob_container_name"],
                upload_file,
            )
            # process the data to vdb
            tax_vdb = text_to_vectordb(
                filename,
                st.session_state["index_name"],
            )
            st.success("Files indexed at Pinecone")
            existing_indices = tax_vdb.get_namespaces()
