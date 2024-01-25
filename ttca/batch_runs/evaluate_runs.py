"""
This compares the accuracy of the winner/tax avoidance questions using RAG vs non-RAG approaches
This is done by comparing the final answers, pulled from langsmith logs.

Author:
Daisy Mak <starrywheat.dm@gmail.com>

Updated on:
21 Jan 2024
"""
from __future__ import annotations

import json

import pandas as pd

from utils_langsmith import get_project_data


def prep_ttca_data(project_name: str, run_type: str) -> pd.DataFrame:
    """Get the batch run data from langsmith and load to a cleaned dataframe"""
    tax_avoidance_runs = get_project_data(project_name)

    data = []
    for run in tax_avoidance_runs:
        row = {}
        # Get run info
        row["RunID"] = run.id
        row["DecisionNumber"] = run.extra["metadata"]["case_number"]
        row["RunTimeStart"] = run.start_time
        row["RunURL"] = run.url
        # row["QueryPrompt"] = run.inputs["question"]

        # Get legal stuff
        try:
            if run_type == "rag":
                response_json = json.loads(run.outputs["answer"])
            elif run_type == "nonrag":
                response_json = json.loads(run.outputs["output_text"])

            def get_value(keyname: str) -> str:
                return (
                    response_json[keyname]
                    if keyname in response_json.keys()
                    else None
                )

            for key in [
                "whetherTaxAvoidanceCase",
                "avoidanceReasoning",
                "winner",
                "winnerReasoning",
            ]:
                value = get_value(key)
                if value is not None:
                    row[key] = value

            data.append(row)
        except:
            pass

    df = pd.DataFrame.from_records(data)
    return df


def read_winner_df(project_name: str, suffix: str, run_type: str):
    cols_needed = ["DecisionNumber", "winner", "winnerReasoning"]
    df = prep_ttca_data(project_name, run_type=run_type)
    df = df[cols_needed]
    df["winner"] = df["winner"].apply(lambda x: x.lower())
    df.rename(
        columns={
            "winner": f"winner_{suffix}",
            "winnerReasoning": f"winnerReasoning_{suffix}",
        },
        inplace=True,
    )
    return df


def read_taxavoidance_df(project_name: str, suffix: str, run_type: str):
    cols_needed = [
        "DecisionNumber",
        "whetherTaxAvoidanceCase",
        "avoidanceReasoning",
    ]
    df = prep_ttca_data(project_name, run_type=run_type)
    df = df[cols_needed]
    df["whetherTaxAvoidanceCase"] = df["whetherTaxAvoidanceCase"].apply(
        lambda x: x.lower(),
    )
    df.rename(
        columns={
            "whetherTaxAvoidanceCase": f"whetherTaxAvoidanceCase_{suffix}",
            "avoidanceReasoning": f"avoidanceReasoning_{suffix}",
        },
        inplace=True,
    )
    return df


# Winner question
df_winner_rag = read_winner_df(
    "ttca_RAG_batch2-winner100-2",
    suffix="rag",
    run_type="rag",
)
df_winner_nonrag_qastuff = read_winner_df(
    "ttca_nonRAG_qastuff_batch100_winner",
    suffix="nonrag_lastchunk",
    run_type="nonrag",
)
df_winner_nonrag_mapreducesum = read_winner_df(
    "ttca_nonRAG_mapreducesum_batch100_winner",
    suffix="nonrag_mapreduce",
    run_type="nonrag",
)

# Combine
df_winner_all = df_winner_rag.merge(
    df_winner_nonrag_qastuff,
    on="DecisionNumber",
)
df_winner_all = df_winner_all.merge(
    df_winner_nonrag_mapreducesum,
    on="DecisionNumber",
)

# Add match column
df_winner_all["RAG vs nonRAG MapReduce match"] = df_winner_all.apply(
    lambda x: len({x["winner_nonrag_mapreduce"], x["winner_rag"]}) < 2,
    axis=1,
)
df_winner_all["RAG vs nonRAG LastChunk match"] = df_winner_all.apply(
    lambda x: len({x["winner_nonrag_lastchunk"], x["winner_rag"]}) < 2,
    axis=1,
)


# Tax avoidance question
df_avoidance_rag = read_taxavoidance_df(
    "ttca_RAG_batch2-taxAvoidance100-2",
    suffix="rag",
    run_type="rag",
)
df_avoidance_nonrag_mapreducesum = read_taxavoidance_df(
    "ttca_nonRAG_mapreducesum_batch100_taxavoidance",
    suffix="nonrag_mapreduce",
    run_type="nonrag",
)

# Combine
df_avoidance_all = df_avoidance_rag.merge(
    df_avoidance_nonrag_mapreducesum,
    on="DecisionNumber",
)

# Add match column
df_avoidance_all["RAG vs nonRAG MapReduce match"] = df_avoidance_all.apply(
    lambda x: len(
        {
            x["whetherTaxAvoidanceCase_nonrag_mapreduce"],
            x["whetherTaxAvoidanceCase_rag"],
        },
    )
    < 2,
    axis=1,
)
