import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from bsc_pathways.sk_model_utils import cross_validate_models


gene_data_path = Path.cwd() / "bsc_pathways/original_genetic_data/data"
genes = pd.read_csv(gene_data_path / "train.csv")
stages = pd.read_csv(gene_data_path / "train_clinical_stage.csv")

genes_test = pd.read_csv(gene_data_path / "test.csv")
stages_test = pd.read_csv(gene_data_path / "test_clinical_stage.csv")

all_patients_genes = stages["sample"].to_list() + stages_test["sample"].to_list()

df_all = pd.read_csv(
    Path.cwd()
    / "bsc_pathways/pathway_enrichment_v2/data/real_patients_enrichments.csv",
    index_col=0,
)

all_patients_pathways = df_all["Patient"].unique().tolist()

ctype_per_patient = {
    patient: c_type
    for patient, c_type in df_all[["Patient", "Cancer_Type"]].drop_duplicates().values
}
gene_expressions = []
initial_stages = []

gene_expressions_lobular = []
initial_stages_lobular = []

gene_expressions_ductal = []
initial_stages_ductal = []


for column_idx, patient_name in enumerate(genes):
    if column_idx != 0:
        if patient_name in all_patients_pathways:
            gene_array = genes[patient_name].values
            stage = stages[stages["sample"] == patient_name][
                "ajcc_pathologic_tumor_stage"
            ].values[0] # type: ignore
            
            if ctype_per_patient[patient_name] == "lobular":
                gene_expressions_lobular.append(gene_array)
                initial_stages_lobular.append(stage)
            elif ctype_per_patient[patient_name] == "ductal":
                gene_expressions_ductal.append(gene_array)
                initial_stages_ductal.append(stage)

for column_idx, patient_name in enumerate(genes_test):
    if column_idx != 0:
        if patient_name in all_patients_pathways:
            gene_array = genes_test[patient_name].values
            stage = stages_test[stages_test["sample"] == patient_name][
                "ajcc_pathologic_tumor_stage"
            ].values[0]  # type: ignore
            
            if ctype_per_patient[patient_name] == "lobular":
                gene_expressions_lobular.append(gene_array)
                initial_stages_lobular.append(stage)
            elif ctype_per_patient[patient_name] == "ductal":
                gene_expressions_ductal.append(gene_array)
                initial_stages_ductal.append(stage)


csv_str = ""

models_to_test = [
    DecisionTreeClassifier(random_state=1),
    RandomForestClassifier(random_state=1),
    GradientBoostingClassifier(random_state=1),
    XGBClassifier(random_state=1),
    SVC(random_state=1),
    MLPClassifier(random_state=1),
]

results_csv_str = cross_validate_models(
    models_list=models_to_test,
    X=np.array(gene_expressions_lobular),
    y=np.array(initial_stages_lobular),
    num_folds=5,
    print_results=True,
)

csv_str += "lobular" + "\n"
csv_str += results_csv_str

models_to_test = [
    DecisionTreeClassifier(random_state=1),
    RandomForestClassifier(random_state=1),
    GradientBoostingClassifier(random_state=1),
    XGBClassifier(random_state=1),
    SVC(random_state=1),
    MLPClassifier(random_state=1),
]

results_csv_str = cross_validate_models(
    models_list=models_to_test,
    X=np.array(gene_expressions_ductal),
    y=np.array(initial_stages_ductal),
    num_folds=5,
    print_results=True,
)

csv_str += "ductal" + "\n"
csv_str += results_csv_str

csv_name = f"gene_stage_classification_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}_separate.csv"  # type: ignore
csv_path = Path.cwd() / "bsc_pathways/original_genetic_data/results" / csv_name
f = open(csv_path, "w")
f.write(csv_str)
f.close()

os.system(f"xdg-open {csv_path}")
