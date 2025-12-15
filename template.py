import os
from  pathlib import Path
import logging
list_of_files = [
    "app.py",
    "Interface/interface.py",
     "Models/classification/svm.py",
      "Models/Regression/linear_regrssor.py",
      "preprocessing/missing_values.py",
      "preprocessing/feature_selection.py",
      "preprocessing/outliers",
     "visalustsation/data_dsitruction.py",
     "visalustsation/feature_relationship.py",

       ]

for filepath in list_of_files :
    filepath=Path(filepath)
    filedir ,filename=os.path.split(filepath)
    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")


    else:
        logging.info(f"{filename} is already exists") 