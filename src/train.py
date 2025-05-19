"""
This module contains functions to preprocess and train the model
for bank consumer churn prediction.
"""
from mlflow.protos.service_pb2 import Dataset
from numpy.random.mtrand import logistic
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.compose import make_column_transformer
import joblib
from sklearn.preprocessing import OneHotEncoder,  StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

### Import MLflow
import mlflow
def rebalance(data):
    """
    Resample data to keep balance between target classes.

    The function uses the resample function to downsample the majority class to match the minority class.

    Args:
        data (pd.DataFrame): DataFrame

    Returns:
        pd.DataFrame): balanced DataFrame
    """
    churn_0 = data[data["Exited"] == 0]
    churn_1 = data[data["Exited"] == 1]
    if len(churn_0) > len(churn_1):
        churn_maj = churn_0
        churn_min = churn_1
    else:
        churn_maj = churn_1
        churn_min = churn_0
    churn_maj_downsample = resample(
        churn_maj, n_samples=len(churn_min), replace=False, random_state=1234
    )

    return pd.concat([churn_maj_downsample, churn_min])


def preprocess(df):
    """
    Preprocess and split data into training and test sets.

    Args:
        df (pd.DataFrame): DataFrame with features and target variables

    Returns:
        ColumnTransformer: ColumnTransformer with scalers and encoders
        pd.DataFrame: training set with transformed features
        pd.DataFrame: test set with transformed features
        pd.Series: training set target
        pd.Series: test set target
    """
    filter_feat = [
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Exited",
    ]
    cat_cols = ["Geography", "Gender"]
    num_cols = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
    ]
    data = df.loc[:, filter_feat]
    data=data.dropna()
    data_bal = rebalance(data=data).reset_index(drop=True)
    X = data_bal.drop("Exited", axis=1)
    y = data_bal["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1912
    )
    col_transf = make_column_transformer(
        (StandardScaler(), num_cols), 
        (OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        remainder="passthrough",
    )

    X_train = col_transf.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=col_transf.get_feature_names_out())

    X_test = col_transf.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=col_transf.get_feature_names_out())

    # Log the transformer as an artifact
    joblib.dump(col_transf, "column_transformer.pkl")
    mlflow.log_artifact("column_transformer.pkl")

    return col_transf, X_train, X_test, y_train, y_test


def train_logistic(max_iter,X_train, y_train):
    """
    Train a logistic regression model.

    Args:
        max_iter : number of iterations
        X_train (pd.DataFrame): DataFrame with features
        y_train (pd.Series): Series with target

    Returns:
        LogisticRegression: trained logistic regression model
    """
    log_reg = LogisticRegression(max_iter=max_iter)
    log_reg.fit(X_train, y_train)

    ### Log the model with the input and output schema
    # Infer signature (input and output schema)
    input_ex = X_train.head(5)
    output_ex= log_reg.predict(input_ex)
    signature=mlflow.models.infer_signature(input_ex,output_ex)
    # Log model
    mlflow.sklearn.log_model(
        log_reg,
        artifact_path="logistic_regression_model", 
        signature=signature,
        input_example=input_ex
        )
    ### Log the data
    dataset = mlflow.data.from_pandas(
            X_train, name="churn_prediction data"
        )
    mlflow.log_input(dataset, context="data")
    return log_reg

def train_random_forest(X_train, y_train):
    """
    Train a random_forest model.

    Args:
        X_train (pd.DataFrame): DataFrame with features
        y_train (pd.Series): Series with target

    Returns:
        RandomForest: trined random_forest model 
    """
    estimtors=200
    mlflow.log_param("n_estimators",estimtors)
    log_forst = RandomForestClassifier(n_estimators=estimtors)

    log_forst.fit(X_train, y_train)

    ### Log the model with the input and output schema
    # Infer signature (input and output schema)
    input_ex = X_train.head(5)
    output_ex= log_forst.predict(input_ex)
    signature=mlflow.models.infer_signature(input_ex,output_ex)
    # Log model
    mlflow.sklearn.log_model(
        log_forst,
        artifact_path="random_forest_model", 
        signature=signature,
        input_example=input_ex
        )
    ### Log the data
    dataset = mlflow.data.from_pandas(
            X_train, name="churn_prediction data"
        )
    mlflow.log_input(dataset, context="data")
    return log_forst
def train_svm(max_iter,X_train,y_train):
    """
    Train a SVM model.

    Args:
        max_iter : number of iterations
        X_train (pd.DataFrame): DataFrame with features
        y_train (pd.Series): Series with target

    Returns:
        SVM: trined SVM model 
    """
    log_svm = svm.SVC(max_iter=max_iter)
    log_svm.fit(X_train, y_train)

    ### Log the model with the input and output schema
    # Infer signature (input and output schema)
    input_ex = X_train.head(5)
    output_ex= log_svm.predict(input_ex)
    signature=mlflow.models.infer_signature(input_ex,output_ex)
    # Log model
    mlflow.sklearn.log_model(
        log_svm,
        artifact_path="SVM_model", 
        signature=signature,
        input_example=input_ex
        )
    ### Log the data
    dataset = mlflow.data.from_pandas(
            X_train, name="churn_prediction data"
        )
    mlflow.log_input(dataset, context="data")
    return log_svm

def mlflow_logging(name):
    with mlflow.start_run(run_name=name) as run:
        run_id = run.info.run_id
        mlflow.set_tag("run_id", run_id)
        df = pd.read_csv("dataset/Churn_Modelling.csv")
        col_transf, X_train, X_test, y_train, y_test = preprocess(df)

        ### Log the max_iter parameter
        
        if name=="logistic_model":
            max_iter=1000
            mlflow.log_param('max_iter',max_iter)
            model=train_logistic(max_iter,X_train,y_train)
        elif name=="Random_forest_model":
            model=train_random_forest(X_train,y_train) 
        elif name=="SVM_model":
            max_iter=1000
            mlflow.log_param('max_iter',max_iter)
            model=train_svm(max_iter,X_train,y_train)
        
        mlflow.set_tag("model_name",name)
        y_pred = model.predict(X_test)
        eval_df = X_test.copy()
        eval_df["Exited"] = y_test.values 
        eval_df["Predictions"] = y_pred

        ### Log metrics after calculating them
        pd_dataset = mlflow.data.from_pandas(
            eval_df, predictions="Predictions", targets="Exited", name="churn Data Evaluation"
        )
        result=mlflow.evaluate(data=pd_dataset, predictions=None, model_type="classifier")
        ### Log tag
        conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
        conf_mat_disp = ConfusionMatrixDisplay(
            confusion_matrix=conf_mat, display_labels=model.classes_
        )
        conf_mat_disp.plot()
        
        # Log the image as an artifact in MLflow
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.show()

def main():
    ### Set the tracking URI for MLflow
    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    ### Set the experiment name
    mlflow.set_experiment("churn_prediction_experiment")
    mlflow_logging("logistic_model")
    mlflow_logging("Random_forest_model")
    mlflow_logging("SVM_model")


if __name__ == "__main__":
    main()
