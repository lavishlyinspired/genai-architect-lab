import os
from pathlib import Path

list_of_files=[
    "QApdf/__init__.py",
    "QApdf/data_ingestion.py",
    "QApdf/embedding.py",
    "QApdf/model_api.py",
    "IPYNBExperiments/experiment.ipynb",
    "StreamlitApp.py",
    "logger.py",
    "exception.py",
    "setup.py"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
      #  print(f"Creating directory: {filedir} for the file: {filename}")
        print(f"creating directory {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) ==0):
        with open(filepath, "w") as f:
            pass