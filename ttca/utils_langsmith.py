"""
Utilities functions to use langsmith

Author:
Daisy Mak <starrywheat.dm@gmail.com>

Updated on:
21 Jan 2024
"""
from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv
from langsmith import Client

load_dotenv()


def get_project_data(project_name: str) -> list:
    """Get the run data of the given project"""
    langsmith_client = Client()
    runs = langsmith_client.list_runs(
        project_name=project_name,
        execution_order=1,
        error=False,
    )
    all_runs = [run for run in runs]
    return all_runs


def add_to_dataset(project_name: str, dataset_name: str):
    """Add the project runs to a dataset"""
    client = Client()
    runs = get_project_data(project_name)
    dataset_name = dataset_name
    if dataset_name not in [x.name for x in client.list_datasets()]:
        dataset = client.create_dataset(
            dataset_name,
            description="RAG approach for gov tax data",
        )
    else:
        dataset = client.read_dataset(dataset_name=dataset_name)
    for run in runs:
        try:
            inputs = run.inputs
            if len(run.tags) > 0:
                inputs["reviewer"] = run.tags[0]
            else:
                inputs["reviewer"] = None
            client.create_example(
                inputs=inputs,
                outputs=run.outputs,
                dataset_id=dataset.id,
                example_id=run.id,
            )
            print(f"Added example {run.id}")
        except ValueError:
            pass
            # print(f"This example {run.id} is added ")
