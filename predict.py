import joblib
import pandas as pd


def load_model(model_path: str):
    """
    load serialize model
    :param model_path:
    :return:
    """
    model = joblib.load(filename=model_path)
    return model


def predict(model, data):
    """
    Generate a prediction of new data
    :param model:
    :param data:
    :return:
    """
    data = pd.DataFrame.from_dict(data)
    prediction = {
        'prediction': model.predict(data)
    }
    return prediction


