from __future__ import annotations

import time

import pinecone
import streamlit as st
import tiktoken
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone as Pinecone_langchain


def tiktoken_len(
    text: str,
    encoding_name: str = "cl100k_base",
    disallowed_special: tuple = (),
) -> int:
    """
    This function returns the number of tokens as used by GPT models,
    for the given text.
    Reference:
    https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

    Args:
        text (str): input text

    Returns:
        int: _description_
    """
    tokenizer = tiktoken.get_encoding(encoding_name)

    # create length function
    tokens = tokenizer.encode(text, disallowed_special=disallowed_special)
    return len(tokens)


def split_documents(
    docs,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks_doc = text_splitter.split_documents(docs)
    return chunks_doc


class PineconeVDB:
    """This is the class to operatate pinecone vectordb.
    This includes init, create, and add documents to the same index. Data are stored on pinecone server.
    This uses Pinecone python client, instead of LangChain vector store object, as this is faster.
    """

    def __init__(
        self,
        index_name: str,
        embed_model_name: str,
        metric: str = "cosine",
    ) -> None:
        self.index_name = index_name
        self.metric = metric
        self.embed_dim = 0
        self.embed_model_name = embed_model_name
        self.text_field = "text"

        # Load the embedding model
        self._load_embed_model()

        # Init the vdb and create a new index if not exist

        self.init_pinecone()
        self.create_pinecone_vdb(self.metric, self.embed_dim)
        # Connect to the index
        self.index = self.connect_pinecone_vdb(self.index_name)

    def _load_embed_model(self) -> None:
        self.embed_model = HuggingFaceEmbeddings(
            model_name=self.embed_model_name,
        )
        self.embed_dim = (
            self.embed_model.client.get_sentence_embedding_dimension()
        )

    def init_pinecone(self) -> None:
        pinecone.init(
            api_key=st.secrets["PINECONE_API_KEY"],
            environment=st.secrets["PINECONE_ENV"],
        )

    def create_pinecone_vdb(
        self,
        metric: str = "cosine",
        embed_dim: int = 768,
    ) -> None:
        if self.index_name not in pinecone.list_indexes():
            # create a new index
            pinecone.create_index(
                name=self.index_name,
                metric=metric,
                dimension=embed_dim,
            )
            # wait for index to finish initialisation
            while not pinecone.describe_index(self.index_name).status["ready"]:
                time.sleep(1)
            print(f"Finish creating {self.index_name} in pinecone.")
        else:
            print(f"{self.index_name} is already created.")

    def connect_pinecone_vdb(self, index_name: str) -> pinecone.Index:
        index = pinecone.Index(index_name)
        # index.describe_index_stats()
        return index

    def add_documents(
        self,
        data,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        overwrite: bool = False,
    ) -> None:
        """
        This function add documents to the index. Each document will be stored in its own namespace
        Assume the input data is list of documents, with the fields: decision_number, content

        Args:
            data : input data
            chunk_size (int, optional): maximum chunk size for splitting. Defaults to 500.
            chunk_overlap (int, optional): overlap size for chunk. Defaults to 20.
        """

        exiting_index = self.get_namespaces()
        for i, record in enumerate(data):
            # create namespace
            namespace = record.metadata["source"].split(".")[0]

            if not overwrite:
                if namespace in exiting_index:
                    print(f"skipping {namespace} indexing")
                    continue
            # get metadata fields: Tax data specific
            metadata = {
                "source": record.metadata["source"],
            }
            # create chunks from content
            splitted_text = split_documents(
                [record],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            # create metadata dicts for each chunk; the splitted text is included in the metadata
            splitted_metadatas = [
                {
                    "chunk": int(j),
                    self.text_field: text.page_content,
                    **metadata,
                }
                for j, text in enumerate(splitted_text)
            ]

            # Insert to index and new namespace
            embeddings = self.embed_model.embed_documents(
                [x.page_content for x in splitted_text],
            )
            ids = [f"{namespace}-{i}" for i in range(len(splitted_text))]
            self.index.upsert(
                vectors=zip(ids, embeddings, splitted_metadatas),
                namespace=namespace,
            )

            # print(self.index.describe_index_stats())
            # print(
            #     f"All {len(data)} texts are embedded and stored in {self.index_name}",
            # )

    def delete_index(self, index_name: str) -> None:
        pinecone.delete_index(index_name)

    def delete_id(self, ids: list[str]) -> None:
        self.index.delete(ids=ids)

    def delete_namespace(self, namespace: str) -> None:
        """This is not supported by gcp-starter environment"""
        self.index.delete(delete_all=True, namespace=namespace)

    def get_namespaces(self) -> list:
        namespaces = [
            x for x in self.index.describe_index_stats().namespaces.keys()
        ]
        namespaces.sort()
        return namespaces

    def get_langchain_pinecone(self) -> Pinecone_langchain:
        # create the langchain wrapper of pinecone
        vectorstore = Pinecone_langchain(
            self.index,
            self.embed_model.embed_query,
            self.text_field,
        )
        return vectorstore
