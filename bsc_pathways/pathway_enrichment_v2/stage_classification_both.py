import os
import numpy as np
import pandas as pd
from pathlib import Path

pd.options.mode.chained_assignment = None  # default='warn'
from datetime import datetime
from collections import Counter
from data_utils import transform_df
from bsc_pathways.sk_model_utils import cross_validate_models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


def get_stages_dataset(df, features):
    features_dict = {"NES": 0, "FDR": 1, "REG": 2}

    stage_labels = []
    pathways = []
    visited_patients = []

    for _, row in df.iterrows():
        if row["Patient"] not in visited_patients:
            pathway_features = [arr[[features_dict[features]]] for arr in row[3:]]

            # if row["Stage"] == 2:
            stage_labels.append(row["Stage"])
            pathways.append(pathway_features)
            visited_patients.append(row["Patient"])
            # else:
            #     stage_labels.append("rest")
            #     pathways.append(pathway_features)
            #     visited_patients.append(row["Patient"])

    return stage_labels, np.squeeze(pathways, axis=2)


if __name__ == "__main__":
    # path of this file
    file_path = os.path.dirname(os.path.realpath(__file__))

    here = Path(__file__).resolve().parent

    # Import the dataframe containing all data (ductal + lobular)
    df_all = pd.read_csv(
        os.path.join(file_path, "data/real_patients_enrichments.csv"),
        index_col=0,
    )

    # transform the dataframe into a more usable representation
    only_one, cancer_type = False, None
    replace_nan, new_value = False, 20
    df = transform_df(
        df_all,
        only_one_type=(only_one, cancer_type),
        replace_nan=(replace_nan, new_value),
        save=False,
    )

    # get stage labels and pathway features from the df
    stage_labels, pathway_features = get_stages_dataset(df, features="NES")

    print(
        "Stage Classification: \t\t",
        f"Labels: {len(stage_labels)}, ",
        f"Features: {pathway_features.shape}, ",
        f"Class support: {dict(Counter(stage_labels))}",
    )

    models_to_test = [
        DecisionTreeClassifier(random_state=1),
        RandomForestClassifier(random_state=1),
        # GradientBoostingClassifier(random_state=1),
        XGBClassifier(random_state=1),
        # SVC(random_state=1),
        # MLPClassifier(random_state=1),
    ]

    results_csv_str = cross_validate_models(
        models_list=models_to_test,
        X=pathway_features,
        y=stage_labels,
        num_folds=5,
        print_results=True,
    )

    csv_name = f"stages_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}_{replace_nan}_{new_value}_all.csv"  # type: ignore
    csv_path = Path.cwd() / "bsc_pathways/pathway_enrichment_v2/results" / csv_name
    f = open(csv_path, "w")
    f.write(results_csv_str)
    f.close()

    os.system(f"xdg-open {csv_path}")
