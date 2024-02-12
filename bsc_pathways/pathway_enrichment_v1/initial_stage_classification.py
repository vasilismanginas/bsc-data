import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

''' 
This file, similarly to original_genetic_data/stage_classification_gene_expressions.py, attempts cancer stage classification.
The differences between the two are:
    1. This file can also use enriched pathways as features (besides gene expressions).
    2. This file has significantly less training data than the other one. This is because it uses trajectory
       starting points as patients, and not all patients are used as starting points in some trajectory.
'''

df = pd.read_csv(
    os.path.join(
        os.getcwd(), f"bsc_pathways/pathway_enrichment_v1/data/lobular_w_negatives.csv"
    )
)

sequence_length = 50
num_pathways = 29
num_expressions = 8954
num_transitions = int(len(df) / sequence_length)

initial_state_labels = []
first_timepoint_gene_expressions = []
first_timepoint_pathways = []

init_stage_from_transition = {
    "I_to_I": 1,
    "I_to_II": 1,
    "II_to_II": 2,
    "II_to_III": 2,
    "III_to_III": 3,
}

# get the first row of each trajectory
# this corresponds to the gene expressions and enriched pathways
# of the starting point of the trajectory, a real patient
for patient_idx in range(num_transitions):
    first_row_index = patient_idx * sequence_length
    first_row_stage = init_stage_from_transition[df.iloc[first_row_index, 1]]
    first_row_genes = df.iloc[first_row_index, 33:].values
    first_row_pathways = df.iloc[first_row_index, 4:33].values

    if first_row_genes.tolist() not in [
        l.tolist() for l in first_timepoint_gene_expressions
    ]:
        initial_state_labels.append(first_row_stage)
        first_timepoint_pathways.append(first_row_pathways)
        first_timepoint_gene_expressions.append(first_row_genes)


X_train, X_test, y_train, y_test = train_test_split(
    np.asarray(first_timepoint_gene_expressions, dtype=float),
    # np.asarray(first_timepoint_pathways, dtype=float),
    np.asarray(initial_state_labels, dtype=float),
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
