import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# path of this file
file_path = os.path.dirname(os.path.realpath(__file__))

# Import the dataframe containing all data (ductal + lobular)
df_all = pd.read_csv(
    os.path.join(file_path, "data/real_patients_enrichments.csv"),
    index_col=0,
)


for cancer_type in ["lobular", "ductal"]:
    # use only patients with this cancer type
    df = df_all[df_all["Cancer_Type"] == cancer_type]

    # keep only the columns of interest
    df = df[["Patient", "Stage", "Pathway", "NES", "FDR q-val", "Regulation"]]

    # convert '---' to NaN and rest of 'NES' column to numeric
    df["NES"] = pd.to_numeric(df["NES"], errors="coerce")

    # sort the dataframe in order to have pathways in the same order across patients
    # reset the index after sorting
    df = df.sort_values(by=["Patient", "Pathway"]).reset_index(drop=True)

    # df.to_csv(os.path.join(file_path, "data", cancer_type + ".csv"), index=False)

    stage_labels = []
    nes_values = []
    reg_values = []
    for row_num in range(0, len(df), 29):
        stage_labels.append(df["Stage"][row_num])
        nes_values.append(df["NES"][row_num : row_num + 29].values)
        reg_values.append(df["Regulation"][row_num : row_num + 29].values)

    X_train, X_test, y_train, y_test = train_test_split(
        nes_values,
        stage_labels,
        train_size=0.75,
        shuffle=True,
    )

    models_to_test = [
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        # because there are NaNs in the data and we haven't concluded how to deal with these
        # for the time being we are limited to classifiers that inherently deal with this
        # SVC(),
        # MLPClassifier(),
    ]

    print("\n", cancer_type)
    for model in models_to_test:
        model.fit(X_train, y_train)
        train_outputs = model.predict(X_train)
        test_outputs = model.predict(X_test)

        print(
            f"{model} - (train) weighted-f1: {f1_score(y_train, train_outputs, average='weighted'):.3f}, (test) weighted-f1: {f1_score(y_test, test_outputs, average='weighted'):.3f}"
        )
