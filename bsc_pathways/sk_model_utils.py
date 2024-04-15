from collections import Counter

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

        if "sklearn" in getattr(model, '__module__', None): # type: ignore
            model.fit(X_train, y_train)
            train_outputs = model.predict(X_train)
            test_outputs = model.predict(X_test)
        else:
            continue

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