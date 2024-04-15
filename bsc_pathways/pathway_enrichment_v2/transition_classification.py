
import os
import numpy as np
import pandas as pd
from pathlib import Path

from pandas.core.indexes.category import contains

pd.options.mode.chained_assignment = None  # default='warn'
from datetime import datetime
from collections import Counter
# from data_utils import transform_df
# from bsc_pathways.sk_model_utils import cross_validate_models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score


def cross_validate_stratified(model, X, y, num_folds=5, print_results=True):
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1)
    skf.get_n_splits(X, y)

    accuracies_train, accuracies_test = [], []
    macro_f1s_train, macro_f1s_test = [], []
    weighted_f1s_train, weighted_f1s_test = [], []

    for _, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index] # type: ignore

        model.fit(X_train, y_train)
        train_outputs = model.predict(X_train)
        test_outputs = model.predict(X_test)

        accuracies_train.append(accuracy_score(y_train, train_outputs))
        accuracies_test.append(accuracy_score(y_test, test_outputs))
        weighted_f1s_train.append(f1_score(y_train, train_outputs, average="weighted"))
        weighted_f1s_test.append(f1_score(y_test, test_outputs, average="weighted"))
        macro_f1s_train.append(f1_score(y_train, train_outputs, average="macro"))
        macro_f1s_test.append(f1_score(y_test, test_outputs, average="macro"))

    metrics = {
        "avg_train_acc": round(sum(accuracies_train) / len(accuracies_train), 4),
        "avg_test_acc": round(sum(accuracies_test) / len(accuracies_test), 4),
        "avg_train_weighted_f1": round(
            sum(weighted_f1s_train) / len(weighted_f1s_train), 4
        ),
        "avg_test_weighted_f1": round(
            sum(weighted_f1s_test) / len(weighted_f1s_test), 4
        ),
        "avg_train_macro_f1": round(sum(macro_f1s_train) / len(macro_f1s_train), 4),
        "avg_test_macro_f1": round(sum(macro_f1s_test) / len(macro_f1s_test), 4),
    }

    if print_results:
        print(model.__class__.__name__)
        print(
            f"( train ) - acc: {metrics['avg_train_acc']}, weighted-f1: {metrics['avg_train_weighted_f1']}, macro-f1: {metrics['avg_train_macro_f1']}",
            f"\t ( test )  - acc: {metrics['avg_test_acc']}, weighted-f1: {metrics['avg_test_weighted_f1']}, macro-f1: {metrics['avg_test_macro_f1']} \n",
        )

    return metrics


def cross_validate_models(
    models_list, X, y, num_folds=5, print_results=True
):
    csv_str = (
        "Class Support,"
        "Classifier,"
        "Accuracy (train),"
        "Weighted-F1 (train),"
        "Macro-F1 (train),"
        "Accuracy (test),"
        "Weighted-F1 (test),"
        "Macro-F1 (test) \n"
    )
    csv_str += f'"{dict(Counter(y))}",'

    for model in models_list:
        metrics = cross_validate_stratified(model, X, y, num_folds, print_results)

        if models_list.index(model) != 0:
            csv_str += ","

        csv_str += (
            f"{model.__class__.__name__},"
            f"{metrics['avg_train_acc']*100},{metrics['avg_train_weighted_f1']*100},{metrics['avg_train_macro_f1']*100},"
            f"{metrics['avg_test_acc']*100},{metrics['avg_test_weighted_f1']*100},{metrics['avg_test_macro_f1']*100}\n"
        )
    
    return csv_str

def transform_df(
    df,
    only_one_type=(False, None),
    replace_nan=(False, None),
    save=False,
):
    if only_one_type[0]:
        # use only patients with this cancer type
        df = df[df["Cancer_Type"] == only_one_type[1]]

    # convert 'NES' column to numeric
    df["NES"] = pd.to_numeric(df["NES"], errors="coerce")

    # choose whether to replace nans and with what value
    if replace_nan[0]:
        df = df.replace(np.nan, replace_nan[1])

    # TrajectoryIDs are given as strings e.g. "[1, 2, 3]"
    # so make this into an actual list of ints -> [1, 2, 3]
    df["TrajectoryID"] = df["TrajectoryID"].apply(
        lambda patient_trajectories: [
            int(traj_id_str.strip())
            for traj_id_str in patient_trajectories[1:-1].split(",")
        ]
    )

    # transform stages to numerical values to use them for sorting later
    stage_dict = {"I": 1, "II": 2, "III": 3}
    df["Stage"] = df["Stage"].apply(lambda s: stage_dict[s])

    # # the list. E.g if row X has TrajectoryID = [1, 2, 3] then the row is deleted and
    # # three copies are created, one where TrajectoryID is 1, one where it is 2, etc.
    # df = df.explode("TrajectoryID")  # EXPLOSION!!! 💣

    # keep only patients that are on one trajectory
    df = df[df['TrajectoryID'].apply(lambda x: len(x) == 1)]

    # create a new column which contains the NES, FDR q-val, and Regulation in one array
    df["Metrics"] = df[["NES", "FDR q-val", "Regulation"]].apply(
        lambda x: np.array(x), axis=1
    )

    # remove these columns as they are all contained within the new Metrics column
    df = df.drop(columns=["Cancer_Type", "NES", "FDR q-val", "Regulation"])

    # pivot the df for a nicer representation
    df_wide = df.pivot_table(
        index=["TrajectoryID", "Stage", "Patient"],
        columns="Pathway",
        values="Metrics",
    ).reset_index()

    # find pathways that contain only NaNs
    # if NaNs have been replaced look for columns which only have the new value
    if replace_nan[0]:
        only_NaN_columns = [
            column_name
            for column_name in df_wide.columns
            if (
                isinstance(df_wide[column_name].dropna().iloc[0], np.ndarray)
                and (np.stack(df_wide[column_name].values) == replace_nan[1]).all()
            )
        ]
    else:
        only_NaN_columns = [
            column_name
            for column_name in df_wide.columns
            if (
                isinstance(df_wide[column_name].dropna().iloc[0], np.ndarray)
                and (np.isnan(np.stack(df_wide[column_name].dropna().values)).all())
            )
        ]

    # drop the columns that are full of NaNs
    df_wide.drop(columns=only_NaN_columns, inplace=True)

    # sort the dataframe in several stages:
    # 1. TrajectoryID: patients with the same ID are placed together
    # 2. Stage: patients are sorted in increasing cancer stage
    df_wide = df_wide.sort_values(by=["TrajectoryID", "Stage"]).reset_index(drop=True)

    # save the new file
    if save:
        df_wide.to_csv(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "data",
                only_one_type[1] + "_trajectories_wide.csv",
            ),
            index=False,
        )

    return df_wide


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

    # transform the dataframe into a more usable representation
    only_one, cancer_type = False, None
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
    ) = get_trajectory_dataset(df, features="NES", invert_concat=True)

    print(
        "Transition Classification: \t",
        f"Labels: {len(transition_labels)}, ",
        f"Features: {concatenation_features.shape}, ",
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
        X=concatenation_features,
        y=transition_labels,
        num_folds=5,
        print_results=True,
    )

    csv_name = f"trajectories_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}_{replace_nan}_{new_value}_all.csv"  # type: ignore
    csv_path = Path.cwd() / "bsc_pathways/pathway_enrichment_v2/results" / csv_name
    f = open(csv_path, "w")
    f.write(results_csv_str)
    f.close()

    os.system(f"xdg-open {csv_path}")