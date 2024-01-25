"""
Classes of RAG and Non-RAG approaches to query TTCA documents
of Tax Avoidance and Winner questions

1. RAG: Standard RAG thats uses pinecone to retrieve the most relevant chunks,
    then rerank using embedding filtering, and finally get reasoning using ChatGPT

2. nonRAG (map-reduce): Break the document into small chunks, get mini-summaries
                    from each chunk using LLM calls (map-process), then get a final
                    summary of the mini-summaries with another LLM call (reduce-process).
                    Finally fed the final-summary to LLM to answer the user query

3. nonRAG (qa): User supply the context, and no retrieval is required, to answer the query.

4. RAG agent: A chatbot style agent that leverage RAG approach to answer user queries.


Author:
Daisy Mak <starrywheat.dm@gmail.com>

Updated on:
22 Nov 2023
"""
from __future__ import annotations

import streamlit as st
import tiktoken
from langchain import PromptTemplate
from langchain.agents import AgentExecutor
from langchain.agents.agent_toolkits import (
    create_conversational_retrieval_agent,
)
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.callbacks import get_openai_callback
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import MessagesPlaceholder
from langchain.pydantic_v1 import BaseModel
from langchain.pydantic_v1 import Field
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema.messages import SystemMessage
from langchain.vectorstores import VectorStore


class TaxAvoidanceResponse(BaseModel):
    whetherTaxAvoidanceCase: str = Field(
        description="Yes/No/Don't know to the Tax avoidance determination",
    )

    avoidanceReasoning: str = Field(
        description="reasoning of the tax avoidance determination",
    )

    # @validator("whetherTaxAvoidanceCase")
    # def answerformat(cls, field):
    #     if not ("Yes" in field or "No" in field or "Don't know") in field:
    #         raise ValueError("Badly formed answer")
    #     return field


class WinnerResponse(BaseModel):
    winner: str = Field(
        description="<HMRC/taxpayer/mixed outcomes/no winner/don’t know>",
    )

    winnerReasoning: str = Field(
        description="summary of the reasoning of the winner",
    )


class RAGpipeline:
    """Base RAG class"""

    def __init__(
        self,
        model_name: str,
        vectordb: VectorStore,
        topk: int = 5,
        temperature: float = 0.0,
    ) -> None:
        self.model_name = model_name
        self.vectordb = vectordb
        self.topk = topk
        self.temperature = temperature
        self._load_llm()

    def _load_llm(self):
        # Chat Model
        self.llm = ChatOpenAI(
            openai_api_key=st.secrets["OPENAI_API_KEY"],
            model_name=self.model_name,
            temperature=self.temperature,
        )

    def _get_retriever(self, namespace: str):
        # Base Retriever
        base_retriever = self.vectordb.as_retriever(
            search_kwargs=dict(k=self.topk, namespace=namespace),
        )

        return base_retriever

    def run_qachain(
        self,
        query: str,
        namespace: str,
        reviewer: str,
    ):
        # QA with source chain
        retriever = self._get_retriever(namespace)
        qa = RetrievalQAWithSourcesChain.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )
        with get_openai_callback() as cb:
            response = qa(
                query,
                tags=[reviewer, namespace],
                metadata={"case_number": namespace, "reviewer": reviewer},
            )
        return response, cb


class RAGpipeline_embfiter(RAGpipeline):
    """RAG with embedding filtering"""

    def __init__(
        self,
        model_name: str,
        embed_model_name: str,
        vectordb: VectorStore,
        topk: int = 5,
        temperature: float = 0.0,
        similarity_threshold: float = 0.5,
    ) -> None:
        super().__init__(model_name, vectordb, topk, temperature)

        self.embed_model_name = embed_model_name
        self.similarity_threshold = similarity_threshold
        self.embeddings = None
        self._load_embeddings()

    def _load_embeddings(self) -> None:
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embed_model_name,
        )

    def _get_retriever(self, namespace: str):
        # Base Retriever
        base_retriever = self.vectordb.as_retriever(
            search_kwargs=dict(k=self.topk, namespace=namespace),
        )

        # Embedding filtering Compressor (rank + filter)
        embeddings_filter = EmbeddingsFilter(
            embeddings=self.embeddings,
            similarity_threshold=self.similarity_threshold,
        )

        # COHERE Reranker
        # cohere_compressor = CohereRerank(cohere_api_key=st.secrets["COHERE_API_KEY"])

        retriever = ContextualCompressionRetriever(
            base_compressor=embeddings_filter,
            base_retriever=base_retriever,
        )

        return retriever


class NonRAGQApipeline:
    """
    Non-RAG approaches to query documents
    1) Map-reduce: use run_sumchain
    2) Specific chunk: use run_qachain

    """

    def __init__(
        self,
        model_name: str,
        question_type: str,
        temperature: float = 0.0,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.question_type = question_type
        self._load_llm()
        self._define_parser()

    def _load_llm(self):
        # Chat Model
        self.llm = ChatOpenAI(
            openai_api_key=st.secrets["OPENAI_API_KEY"],
            model_name=self.model_name,
            temperature=self.temperature,
        )

    def _define_parser(self):
        self.tax_avoidance_output_parser = PydanticOutputParser(
            pydantic_object=TaxAvoidanceResponse,
        )

        self.winner_output_parser = PydanticOutputParser(
            pydantic_object=WinnerResponse,
        )

    def _prompt_template(self, query: str) -> None:
        if self.question_type == "tax_avoidance":
            format_instructions = (
                self.tax_avoidance_output_parser.get_format_instructions()
            )
        elif self.question_type == "winner":
            format_instructions = (
                self.winner_output_parser.get_format_instructions()
            )

        self.map_prompt = PromptTemplate(
            template="""
            Write a concise summary of the following:

            "{text}"

            CONCISE SUMMARY:

            """,
            input_variables=["text"],
        )

        self.combine_prompt = PromptTemplate(
            template="""
            You are a UK tax expert. Given the following extracted parts of a judgment of the
            First Tier Tribunal (formally the Special Commissioners), delimited by triple backquotes,
            analyze the context provided and respond to determine if this case centers on tax avoidance.
            First Tier Tribunal is the first instance tax appeal body from decisions of HMRC. The appellant is always the taxpayer.

            {query}

            {format_instructions}

            EXTRACTED PARTS:
            ```{text}```

            FINAL ANSWER:
            """,
            input_variables=["text"],
            partial_variables={
                "query": query,
                "format_instructions": format_instructions,
            },
        )

        self.question_prompt = PromptTemplate(
            template="""
            You are a UK tax expert. The context, delimited by triple backquotes, is a judgment of the First Tier Tribunal (formally the Special Commissioners) which is the first instance tax appeal body from decisions of HMRC. The appellant is always the taxpayer.
            Answer the question as precise as possible using the provided context.

            {format_instructions}

            CONTEXT:
            ```{context}```

            QUESTION:
            {question}

            FINAL ANSWER:
            """,
            input_variables=["context", "question"],
            partial_variables={
                "format_instructions": format_instructions,
            },
        )

    def count_tokens(self, context: str) -> int:
        encoding = tiktoken.encoding_for_model(self.model_name)
        tokens = encoding.encode(context)
        return len(tokens)

    def run_sumchain(
        self,
        query: str,
        context,
        namespace: str,
        reviewer: str,
    ):
        # Setup the prompts
        self._prompt_template(query)

        # Create summary of each chunk, then ask the question on the combined
        qasum = load_summarize_chain(
            self.llm,
            chain_type="map_reduce",
            map_prompt=self.map_prompt,
            combine_prompt=self.combine_prompt,
        )
        with get_openai_callback() as cb:
            response = qasum(
                context,
                tags=[reviewer, namespace],
                metadata={"case_number": namespace, "reviewer": reviewer},
            )

        if self.question_type == "tax_avoidance":
            parsed_response = self.tax_avoidance_output_parser.parse(
                response["output_text"],
            )
        elif self.question_type == "winner":
            parsed_response = self.winner_output_parser.parse(
                response["output_text"],
            )
        return parsed_response, cb

    def run_qachain(
        self,
        query: str,
        context,
        namespace: str,
        reviewer: str,
    ):
        # QA with source chain
        qa = load_qa_chain(
            self.llm,
            chain_type="stuff",
            prompt=self.question_prompt,
        )

        with get_openai_callback() as cb:
            response = qa(
                {
                    "input_documents": context,
                    "question": query,
                },
                tags=[reviewer, namespace],
                metadata={"case_number": namespace, "reviewer": reviewer},
            )
        parsed_response = self.winner_output_parser.parse(
            response["output_text"],
        )
        return parsed_response, cb


def agent_rag_pipeline(
    model_name: str,
    vectordb: VectorStore,
    namespace: str,
    topk: int = 5,
    temperature: float = 0.0,
) -> AgentExecutor:
    """
    Agent to run RAG approach

    Args:
        model_name (str): Name of OpenAI model
        vectordb (VectorStore): vector db object
        namespace (str): namespace of the vectordb
        topk (int, optional): Number of top documents to retrieve. Defaults to 5.
        temperature (float, optional): LLM temperature. Defaults to 0.0.

    Returns:
        AgentExecutor: RAG agent
    """
    # Chat Model
    llm = ChatOpenAI(
        openai_api_key=st.secrets["OPENAI_API_KEY"],
        model_name=model_name,
        temperature=temperature,
    )

    # Retriever
    retriever = vectordb.as_retriever(
        search_kwargs=dict(k=topk, namespace=namespace),
    )

    # Compressor
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever,
    )

    # Retriever Tool
    ret_tool = create_retriever_tool(
        compression_retriever,
        "search_documents",
        "Providing information and good for answering questions about tax tribunal case documents",
    )
    tools = [ret_tool]

    # System Prompt
    system_message = SystemMessage(
        content=(
            """Do your best to answer the questions.
        Feel free to use any tools available to look up relevant information, only if neccessary.
        """
        ),
    )
    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=system_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name="history")],
    )
    agent_executor = create_conversational_retrieval_agent(
        llm,
        tools,
        verbose=True,
        prompt=prompt,
    )

    return agent_executor
