from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

def load_data(test_size=0.2, random_state=42):
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    accuracy = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, f1

def save_model(model, file_path="model.pkl"):
    joblib.dump(model, file_path)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)
    save_model(model)
    accuracy, f1 = evaluate_model(model, X_test, y_test)
    print(f"Model accuracy: {accuracy}")
    print(f"Model F1 score: {f1}")
