"""
Use this script to run the

1. Batch index of all documents on Azure Blob
2. Extraction of the questions on all TTCA data, with different RAG/non-RAG approaches

For Tax avoidance, we run RAG and nonRAG (map reduce)
For Winner, we run RAG and nonRAG (last chunk)

Author:
Daisy Mak <starrywheat.dm@gmail.com>

Updated on:
22 November 2023

"""
from __future__ import annotations

import json
import os

import pandas as pd
from langchain.output_parsers import PydanticOutputParser
from pdf2image.exceptions import PDFInfoNotInstalledError
from pinecone.core.client.exceptions import ApiException
from prep_doc import read_doc_from_blob
from prep_doc import text_to_vectordb
from tqdm import tqdm
from utils_llm import NonRAGQApipeline
from utils_llm import RAGpipeline_embfiter
from utils_llm import TaxAvoidanceResponse
from utils_llm import WinnerResponse
from vectordb import PineconeVDB
from vectordb import split_documents


def batch_run(
    caselist: list,
    langsmith_project_name: str,
    question_prompt: str,
) -> None:
    """
    Run the RAG pipeline on a batch
    """
    os.environ["LANGCHAIN_PROJECT"] = langsmith_project_name
    # load vectordb
    tax_vdb = PineconeVDB(
        INDEX_NAME,
        VDB_EMBED_MODEL_NAME,
    )

    # load QA pipeline
    ragpipeline = RAGpipeline_embfiter(
        CHATMODEL_NAME,
        RETRIEVER_EMBED_MODEL_NAME,
        tax_vdb.get_langchain_pinecone(),
        topk=TOP_K,
        temperature=TEMPERATURE,
        similarity_threshold=SIMILARITY_THRESHOLD,
    )

    for ix, decisionnumber in tqdm(enumerate(caselist)):
        result, cb = ragpipeline.run_qachain(
            question_prompt,
            decisionnumber,
            REVIEWER,
        )
        print(f'Decision Number: {ix} {decisionnumber}\n{result["answer"]}')
        print(f"Cost = USD{round(cb.total_cost,4)}")
        # time.sleep(5)


def existing_data_index():
    pinecone_index_name = "tax-vdb"
    tax_vdb = PineconeVDB(
        pinecone_index_name,
        "BAAI/bge-base-en",
        metric="cosine",
    )
    """ Get the case number in vectordb index"""
    existing_index = tax_vdb.get_namespaces()
    return existing_index


def batch_index(df_batch: pd.DataFrame):
    """Calculate the vector embedding and store to Pinecone"""
    existing_index = existing_data_index()
    container_name = "casedocuments"
    for ids, row in tqdm(df_batch.iterrows()):
        # Get the case info
        decisionnumber = row["DecisionNumber"]
        url = row["DecisionURL"]
        file_extension = url.split(".")[-1].lower()
        filename = f"{decisionnumber.upper()}.{file_extension}"
        try:
            if decisionnumber not in existing_index:
                _ = text_to_vectordb(
                    filename,
                    existing_index,
                    container_name=container_name,
                )
                print(f"Added {decisionnumber} to tax-vdb")
            else:
                print(f"Skipping {decisionnumber}")
        except ApiException:
            print(f"Can't index {decisionnumber}")
        except PDFInfoNotInstalledError:
            print(f"Problems with {decisionnumber} pdf")


def winner_nonrag(
    df_batch: pd.DataFrame,
    langsmith_project_name: str,
) -> None:
    """Run the nonRAG for winner question"""
    # Settings
    blob_container_name = "casedocuments"
    nonrag = NonRAGQApipeline("gpt-3.5-turbo-1106", "winner")
    os.environ["LANGCHAIN_PROJECT"] = langsmith_project_name
    query = """
        Review the judgment to determine the predominant winner of the case, which is either HMRC or the taxpayer.
        First, search the case for the following terms:
        - “the appeal is upheld” or the “appeal is allowed" or similar formulations. An appeal that is upheld or allowed usually indicates the taxpayer is the winner.
        - “the appeal is dismissed” or “cannot allow the appeal” or similar formulations. An appeal that is dismissed typically means HMRC is the winner.
        - “the appeal is partially allowed" or similar formulations. "Partially Allowed" suggests mixed outcomes. Please respond “mixed outcomes” if this is the case.
        """

    # Load each case in batch
    for ids, row in tqdm(df_batch.iterrows()):
        # Get the case info
        decisionnumber = row["DecisionNumber"]
        url = row["DecisionURL"]
        file_extension = url.split(".")[-1].lower()
        filename = f"{decisionnumber.upper()}.{file_extension}"
        doc = read_doc_from_blob(
            container_name=blob_container_name,
            blob_name=filename,
        )

        # Use last paragraphs of document
        splitted_context = split_documents(
            doc,
            chunk_size=5096,  # 512
            chunk_overlap=32,
        )

        context = splitted_context[-2:]

        result, cb = nonrag.run_sumchain(
            query,
            context,
            decisionnumber,
            REVIEWER,
        )
        print(f"Decision Number: {decisionnumber}\n{result}")
        print(f"Cost = USD{round(cb.total_cost,4)}")


def tax_avoidance_nonrag(
    df_batch: pd.DataFrame,
    langsmith_project_name: str,
) -> None:
    """Run the nonRAG for tax avoidance question"""
    # Settings
    blob_container_name = "casedocuments"
    nonrag = NonRAGQApipeline("gpt-3.5-turbo-1106", "tax_avoidance")
    os.environ["LANGCHAIN_PROJECT"] = langsmith_project_name
    query = """
        Tax avoidance refers to the use of legal methods to minimize the amount of income tax owed by an individual or a business.
        Note that the term 'tax avoidance' or its variations will frequently appear in
        judgments focused on this issue. However, it's crucial to recognize when the term is
        used in a context that doesn't imply tax avoidance, such as "there is no suggestion of tax avoidance".
    """

    # Load each case in batch
    for ids, row in tqdm(df_batch.iterrows()):
        # Get the case info
        decisionnumber = row["DecisionNumber"]
        url = row["DecisionURL"]
        file_extension = url.split(".")[-1].lower()
        filename = f"{decisionnumber.upper()}.{file_extension}"
        doc = read_doc_from_blob(
            container_name=blob_container_name,
            blob_name=filename,
        )
        context = split_documents(
            doc,
            chunk_size=5096,
            chunk_overlap=32,
        )
        # context = doc

        # Run QA
        result, cb = nonrag.run_sumchain(
            query,
            context,
            decisionnumber,
            REVIEWER,
        )
        print(f"Decision Number: {decisionnumber}\n{result}")
        print(f"Cost = USD{round(cb.total_cost,4)}")


if __name__ == "__main__":
    # Settings
    INDEX_NAME = "tax-vdb"
    VDB_EMBED_MODEL_NAME = "BAAI/bge-base-en"
    CHATMODEL_NAME = "gpt-3.5-turbo-1106"
    RETRIEVER_EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
    TOP_K = 5
    TEMPERATURE = 0.0
    SIMILARITY_THRESHOLD = 0.1
    REVIEWER = "Daisy"

    # Get the case number to run in batch mode
    df = pd.read_csv(
        "/Users/daisymak/Projects/gov_tax_data/data/case_documents/ttca-export-171023-1.csv",
    )

    # Uncomment line 221 to batch index in pinecone
    # batch_index(df)

    #  Configure the tax avoidance and winner questions settings
    ta_projectname = "ttca_RAG_batch2-taxAvoidance100-duplicate_check"
    winner_projectname = "ttca_RAG_batch8k_winner"
    tax_avoidance_output_parser = PydanticOutputParser(
        pydantic_object=TaxAvoidanceResponse,
    )
    winner_output_parser = PydanticOutputParser(
        pydantic_object=WinnerResponse,
    )

    # Get all the cases in vectorDB index
    caselist = existing_data_index()

    # Uncomment line 238-258 to run RAG
    # QUESTION_PROMPT = """
    # You are a UK tax expert. The context is a judgment of the First Tier Tribunal (formally the Special Commissioners) which is the first instance tax appeal body from decisions of HMRC. The appellant is always the taxpayer. Respond to the following question. Your responses should be structured in JSON format.

    # Review the judgment to determine the predominant winner of the case, which is either HMRC or the taxpayer.
    # First, search the case for the following terms:
    # - “the appeal is upheld” or the “appeal is allowed" or similar formulations. An appeal that is upheld or allowed usually indicates the taxpayer is the winner.
    # - “the appeal is dismissed” or “cannot allow the appeal” or similar formulations. An appeal that is dismissed typically means HMRC is the winner.
    # - “the appeal is partially allowed" or similar formulations. "Partially Allowed" suggests mixed outcomes. Please respond “mixed outcomes” if this is the case.

    # Your response format:
    # {
    #     "winner": "<HMRC/taxpayer/mixed outcomes/do not know>",
    #     "winnerReasoning": "<summary>"
    # }

    # FOR ILLUSTRATION PURPOSES ONLY:
    # Here is an example of how you should structure your output based on a hypothetical case:
    # {“winner”: “HMRC”, “winnerReasoning”: “the tribunal found the payment was an annual payment and therefore taxable. The appeal was therefore dismissed.”}
    # Your actual response should be based on the specifics of the judgment context provided and not mirror the example.  """
    # batch_run(caselist, winner_projectname)
    # batch_run(caselist, ta_projectname)

    # Uncomment line 263 to run nonRAG for TA question
    # tax_avoidance_nonrag(df, ta_projectname)

    # Uncomment line 265 to run nonRAG for winner question
    # winner_nonrag(df, winner_projectname)
