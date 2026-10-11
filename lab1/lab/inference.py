import joblib

class_map = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}

def load_model(file_path="model.pkl"):
    return joblib.load(file_path)

def predict(model, input_data):
    input_data_array = [list(input_data.values())]
    prediction = model.predict(input_data_array)
    return class_map[prediction[0]]
