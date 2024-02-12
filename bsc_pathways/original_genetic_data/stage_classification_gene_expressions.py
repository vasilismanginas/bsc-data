import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


gene_data_path = os.path.join(os.getcwd(), f"bsc_pathways/original_genetic_data/data")
genes = pd.read_csv(os.path.join(gene_data_path, "train.csv"))
stages = pd.read_csv(os.path.join(gene_data_path, "train_clinical_stage.csv"))

gene_expressions = []
initial_stages = []

for column_idx, patient_name in enumerate(genes):
    if column_idx != 0:
        gene_array = genes[patient_name].values
        stage = stages[stages["sample"] == patient_name][
            "ajcc_pathologic_tumor_stage"
        ].values[0] # type: ignore

        gene_expressions.append(gene_array)
        initial_stages.append(stage)

X_train, X_test, y_train, y_test = train_test_split(
    np.asarray(gene_expressions),
    np.asarray(initial_stages),
    test_size=0.2,
    shuffle=True,
)

models_to_test = [
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    SVC(),
    MLPClassifier(),
]

for model in models_to_test:
    model.fit(X_train, y_train)
    test_outputs = model.predict(X_test)

    print(model)
    print(classification_report(y_test, test_outputs))
    print("\n\n")

print()
