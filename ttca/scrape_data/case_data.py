from __future__ import annotations

import csv
import os

import numpy as np
import pandas as pd
import requests

# read case data
df = pd.read_csv(
    "/Users/daisymak/Projects/gov_tax_data/data/case_documents/ttca_case_table_full.csv",
)

# download the case document
docfilepath = os.path.join(
    "/Users/daisymak/Projects/gov_tax_data/data/case_documents/all",
)
for ids, row in df.iterrows():
    # Get the case info
    decisionnumber = row["DecisionNumber"]
    if decisionnumber != np.nan:
        url = row["DecisionURL"]
        file_extension = url.split(".")[-1].lower()
        filename = f"{decisionnumber.upper()}.{file_extension}"
        try:
            # Download the file
            r = requests.get(url)

            # Save the file
            with open(os.path.join(docfilepath, filename), "wb") as f:
                f.write(r.content)

            print(f"{decisionnumber}: written to {filename} ({url})")
        except:
            print(f"skipping {filename}")
