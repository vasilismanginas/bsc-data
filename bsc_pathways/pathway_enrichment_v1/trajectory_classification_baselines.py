import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


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

trajectory_labels = np.empty([num_transitions], dtype=object)
first_last_pathway_concat = np.empty([num_transitions, num_pathways * 2])
first_last_genes_concat = np.empty([num_transitions, num_expressions * 2])
first_last_pathway_difference = np.empty([num_transitions, num_pathways])
first_last_genes_difference = np.empty([num_transitions, num_expressions])

for patient_idx in range(num_transitions):
    first_row_index = patient_idx * sequence_length
    last_row_index = (patient_idx + 1) * sequence_length - 1
    trajectory_label = df.iloc[first_row_index, 1]

    first_row_genes = df.iloc[first_row_index, 33:].values
    last_row_genes = df.iloc[last_row_index, 33:].values

    first_row_pathways = df.iloc[first_row_index, 4:33].values
    last_row_pathways = df.iloc[last_row_index, 4:33].values

    first_row = df.iloc[first_row_index, 4:33].values
    last_row = df.iloc[last_row_index, 4:33].values

    trajectory_labels[patient_idx] = df.iloc[first_row_index, 1]
    first_last_genes_concat[patient_idx] = np.concatenate(
        (first_row_genes, last_row_genes)
    )
    first_last_genes_difference[patient_idx] = np.subtract(
        last_row_genes, first_row_genes
    )
    first_last_pathway_concat[patient_idx] = np.concatenate(
        (first_row_pathways, last_row_pathways)
    )
    first_last_pathway_difference[patient_idx] = np.subtract(
        last_row_pathways, first_row_pathways
    )


# here we choose what features we use
# we use either gene expressions or pathways and either the
# concatenation of the first and last row or their difference
X_train, X_test, y_train, y_test = train_test_split(
    # first_last_genes_concat,
    # first_last_genes_difference,
    # first_last_pathway_concat,
    first_last_pathway_difference,
    trajectory_labels,
    test_size=0.20,
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
