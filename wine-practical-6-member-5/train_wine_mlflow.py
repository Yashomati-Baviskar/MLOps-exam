import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("wine-classification-practical-6")

wine = load_wine(as_frame=True)
X = wine.data.rename(columns={"od280/od315_of_diluted_wines": "od280_od315_diluted_wines"})
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Run 1: Logistic Regression
with mlflow.start_run(run_name="logistic_regression_baseline"):
    c_val = 1.0
    max_iter = 500
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("C", c_val)
    mlflow.log_param("max_iter", max_iter)

    clf = LogisticRegression(C=c_val, max_iter=max_iter, random_state=42)
    clf.fit(X_train_scaled, y_train)
    preds = clf.predict(X_test_scaled)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")
    prec = precision_score(y_test, preds, average="weighted")
    rec = recall_score(y_test, preds, average="weighted")

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_weighted", f1)
    mlflow.log_metric("precision_weighted", prec)
    mlflow.log_metric("recall_weighted", rec)
    mlflow.sklearn.log_model(clf, name="model")

    print(f"LogisticRegression -> accuracy: {acc:.4f}, f1: {f1:.4f}")

# Run 2 & 3: Random Forest hyperparameter tuning
for n_estimators, max_depth in [(50, 4), (150, 8)]:
    run_name = f"random_forest_n{n_estimators}_d{max_depth}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
        )
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        prec = precision_score(y_test, preds, average="weighted")
        rec = recall_score(y_test, preds, average="weighted")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_weighted", f1)
        mlflow.log_metric("precision_weighted", prec)
        mlflow.log_metric("recall_weighted", rec)
        mlflow.sklearn.log_model(
            rf,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        print(f"{run_name} -> accuracy: {acc:.4f}, f1: {f1:.4f}")