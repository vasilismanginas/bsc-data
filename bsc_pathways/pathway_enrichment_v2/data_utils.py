import os
import numpy as np
import pandas as pd


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

    # every row in which TrajectoryID is list-like is duplicated for each element in
    # the list. E.g if row X has TrajectoryID = [1, 2, 3] then the row is deleted and
    # three copies are created, one where TrajectoryID is 1, one where it is 2, etc.
    df = df.explode("TrajectoryID")  # EXPLOSION!!!

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
