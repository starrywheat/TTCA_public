"""
Utility functions to

1. read and load local files,
2. upload/download to Azure Blob
3. Index Azure blobs to VectorDB


Author:
Daisy Mak <starrywheat.dm@gmail.com>

Updated on:
21 Jan 2024
"""
from __future__ import annotations

import os
import zipfile

import streamlit as st
from azure.storage.blob import BlobServiceClient
from langchain.document_loaders import AzureBlobStorageContainerLoader
from langchain.document_loaders import AzureBlobStorageFileLoader
from langchain.document_loaders import TextLoader
from vectordb import PineconeVDB


def read_txt(filepath: str) -> str:
    """Read single local document"""
    txt_data = TextLoader(filepath)
    return txt_data.load()[0].page_content


def read_docs(folder_name: str) -> list:
    """Read local documents from a directory"""
    data = []

    for f in os.listdir(folder_name):
        if f.endswith(".txt"):
            text = read_txt(os.path.join(folder_name, f))
            data.append(
                {"decision_number": f.replace(".txt", ""), "content": text},
            )
    return data


def write_to_file(filename: str, data: bytes):
    """unzip files from .zip and write to local"""
    folder = "data"
    if not os.path.exists(folder):
        os.mkdir(folder)
    filepath = os.path.join(folder, filename)
    with open(filepath, "wb") as f:
        f.write(data)

    # if this is zip file, unzip the files
    if filename.endswith(".zip"):
        zip_file = zipfile.ZipFile(filepath)
        folder = os.path.join(folder, filename.replace(".zip", ""))
        zip_file.extractall(folder)

    return folder


def doc2pdf_linux(filename):
    """Convert docx to pdf"""
    cmd = f"soffice --headless --convert-to pdf --outdir data {filename}"
    os.system(cmd)
    newfilename = filename.split(".")[0] + ".pdf"
    return newfilename


def get_blob_client(container_name: str, blob_name: str):
    """Get Azure blob client"""
    blob_service_client = BlobServiceClient(
        st.secrets["AZURE_BLOB_ENDPOINT"],
        st.secrets["AZURE_BLOB_CREDS"],
    )
    blob_client = blob_service_client.get_blob_client(
        container=container_name,
        blob=blob_name,
    )
    return blob_client


def download_blob(container_name: str, blob_name: str):
    """Download blob as binary"""
    blob_client = get_blob_client(container_name, blob_name)
    b = blob_client.download_blob().readall()
    return b


def upload_blob_data(container_name: str, uploadfile):
    """Upload local file to Azure blob"""
    # Convert doc to pdf before upload
    if uploadfile.name.lower().endswith(
        ".doc",
    ) or uploadfile.name.lower().endswith(".docx"):
        write_to_file(uploadfile.name, uploadfile.getvalue())
        filename = doc2pdf_linux(os.path.join("data", uploadfile.name))
        with open(filename, "rb") as f:
            data = f.read()
        filename = filename.split("/")[-1]
    else:
        filename = uploadfile.name
        data = uploadfile.getvalue()

    # Upload
    blob_client = get_blob_client(container_name, filename)

    blob_client.upload_blob(data, blob_type="BlockBlob", overwrite=True)
    print(f"File {filename} uploaded to {container_name}")
    return filename


def read_doc_from_blob(container_name: str, blob_name: str) -> list:
    """Read a single document from Azure Blob"""
    loader = AzureBlobStorageFileLoader(
        conn_str=st.secrets["AZURE_BLOB_CONN_STR"],
        container=container_name,
        blob_name=blob_name,
    )

    documents = loader.load()
    for doc in documents:
        source = doc.metadata["source"]
        doc.metadata["source"] = source.split("/")[-1]

    return documents


def read_alldocs_from_blob(container_name: str) -> list:
    """Read all documents in Azure Blob container"""
    loader = AzureBlobStorageContainerLoader(
        conn_str=st.secrets["AZURE_BLOB_CONN_STR"],
        container=container_name,
    )
    documents = loader.load()

    # Rename the filename
    for doc in documents:
        source = doc.metadata["source"]
        doc.metadata["source"] = source.split("/")[-1]
    return documents


def text_to_vectordb(
    filename: str,
    pinecone_index_name: str,
    container_name: str = "tax-tribunal",
    embed_model_name: str = "BAAI/bge-base-en",
    metric: str = "cosine",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    overwrite: bool = True,
) -> PineconeVDB:
    """
    Index documents stored in Azure blob to Pinecone DB

    Args:
        filename (str): Filename. If left "", all files found on the container will be indexed
        pinecone_index_name (str): pinecone index name
        container_name (str, optional): Azure blob container name. Defaults to "tax-tribunal".
        embed_model_name (str, optional): Name of embedding model for use in indexing. Defaults to "BAAI/bge-base-en".
        metric (str, optional): VectorDB similarity metric name. Defaults to "cosine".
        chunk_size (int, optional): Chunk size. Defaults to 512.
        chunk_overlap (int, optional): Size of overlapping chunk. Defaults to 64.
        overwrite (bool, optional): Overwrite to vectordb. Defaults to True.

    Returns:
        PineconeVDB: VectorDB
    """
    # Create the pinecone vdb
    tax_vdb = PineconeVDB(
        pinecone_index_name,
        embed_model_name,
        metric=metric,
    )

    # Add the documents
    if filename != "":
        tax_data = read_doc_from_blob(
            container_name=container_name,
            blob_name=filename,
        )

    else:  # Load all in blob storage
        tax_data = read_alldocs_from_blob(container_name)

    tax_vdb.add_documents(
        tax_data,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        overwrite=overwrite,
    )

    return tax_vdb
