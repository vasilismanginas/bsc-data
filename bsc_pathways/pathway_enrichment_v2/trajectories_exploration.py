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

def get_trajectory_dataset(df, features, invert_concat):
    features_dict = {"NES": 0, "FDR": 1, "REG": 2}

    concatenation_features = []
    difference_features = []
    transition_labels = []

    # loop over all trajectoryIDs
    for traj_id in range(df["TrajectoryID"][0], df["TrajectoryID"].max() + 1):
        this_trajectory = df[df["TrajectoryID"] == traj_id]

        # break trajectory into pairs of stages (I to II, II to III)
        for idx in range(len(this_trajectory) - 1):
            two_stages = this_trajectory[idx : idx + 2]

            transition_label = (
                str(two_stages["Stage"].iloc[0])
                + "_to_"
                + str(two_stages["Stage"].iloc[1])
            )

            # get the features for the initial and final stages. Features are either
            # the NES, FDR q-val, or Regulation value for each of pathways. So this
            # gives two arrays with 29 values (actually less bc some pathways removed).
            initial_features, final_features = list(
                map(
                    lambda stage: [arr[features_dict[features]] for arr in stage],
                    two_stages.values[:, 3:],
                )
            )

            # calculate the concatenation of the initial and final stages
            # as well as their difference
            if invert_concat:
                if transition_label == "1_to_2":
                    concatenation = np.array(initial_features + final_features)
                elif transition_label == "2_to_3":
                    concatenation = np.array(final_features + initial_features)
                    transition_label = "3_to_2"
            else:
                concatenation = np.array(initial_features + final_features)
            
            difference = np.array(initial_features) - np.array(final_features)

            transition_labels.append(transition_label)
            concatenation_features.append(concatenation) # type: ignore
            difference_features.append(difference) # type: ignore

    return (
        transition_labels,
        np.array(concatenation_features),
        np.array(difference_features),
    )


if __name__ == "__main__":
    # path of this file
    file_path = os.path.dirname(os.path.realpath(__file__))

    # Import the dataframe containing all data (ductal + lobular)
    df_all = pd.read_csv(
        os.path.join(file_path, "data/real_patients_enrichments.csv"),
        index_col=0,
    )

    csv_str = ""
    for cancer_type in ["lobular", "ductal"]:
        # transform the dataframe into a more usable representation
        only_one, cancer_type = True, cancer_type
        replace_nan, new_value = False, None
        df = transform_df(
            df_all,
            only_one_type=(only_one, cancer_type),
            replace_nan=(replace_nan, new_value),
            save=False,
        )

        (
            transition_labels,
            concatenation_features,
            difference_features,
        ) = get_trajectory_dataset(df, features="NES", invert_concat=False)

        print(
            "Transition Classification: \t",
            f"Labels: {len(transition_labels)}, ",
            f"Features: {difference_features.shape}, ",
            f"Class support: {dict(Counter(transition_labels))} \n"
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
            X=difference_features,
            y=transition_labels,
            num_folds=5,
            print_results=True,
        )

        csv_str += cancer_type + "\n"
        csv_str += results_csv_str


    csv_name = f"trajectories_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}_{replace_nan}_{new_value}.csv"  # type: ignore
    csv_path = Path.cwd() / "bsc_pathways/pathway_enrichment_v2/results" / csv_name
    f = open(csv_path, "w")
    f.write(csv_str)
    f.close()

    os.system(f"xdg-open {csv_path}")