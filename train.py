import sys
import json
import joblib
import logging
import warnings
from split_data import load_data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import GradientBoostingClassifier
from utilities.evaluation import calculate_metrics, metrics_summary
from utilities.custom_pipeline import ColumnSelector, ConvertDtypes, \
    GetDummies, GetDataFrame, BooleanTransformation
from utilities.config import feature_columns_dtypes, label_column_dtype, to_boolean, label_column, \
    numerical_features, categorical_features, cols_to_modeling, final_features_to_modeling

warnings.filterwarnings(action='ignore')

logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
console_handle = logging.StreamHandler(sys.stdout)
console_handle.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s -%(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console_handle.setFormatter(formatter)
logger.addHandler(console_handle)


def define_preprocessing_pipeline():
    """

    :return:
    """
    general_transformations = Pipeline([('boolean', BooleanTransformation(columns=to_boolean)),
                                        ('dtypes', ConvertDtypes(numerical=numerical_features,
                                                                 categorical=categorical_features)),
                                        ('for_modeling', ColumnSelector(columns=cols_to_modeling))])
    numerical_transformations = Pipeline([('numerical_selector', ColumnSelector(columns=numerical_features)),
                                          ('scaler', StandardScaler()),
                                          ('numerical_df', GetDataFrame(columns=numerical_features))])
    categorical_transformations = Pipeline([('categorical_selector', ColumnSelector(columns=categorical_features)),
                                            ('ohe', GetDummies(columns=categorical_features))])
    preprocessor = Pipeline([('general', general_transformations),
                             ('features', FeatureUnion([
                                 ('numerical', numerical_transformations),
                                 ('categorical', categorical_transformations)
                             ])),
                             ('final_df', GetDataFrame(columns=final_features_to_modeling))])
    logger.info('El Pipeline de preprocesamiento se ha creado correctamente.')
    return preprocessor


def train(transformer_pipeline, train_data):
    """

    :param transformer_pipeline:
    :param train_data:
    :return:
    """
    params = {
        'estimator__learning_rate': [0.05, 0.1],
        'estimator__max_depth': [3, 5, 7],
        'estimator__subsample': [0.7, 1.0]
    }
    train_label = train_data.pop(label_column)
    gb = Pipeline([('preprocessor', transformer_pipeline),
                   ('estimator', GradientBoostingClassifier(random_state=42))])
    model = GridSearchCV(estimator=gb, param_grid=params, scoring='f1', cv=5).fit(train_data, train_label)
    logger.info('El modelo ha sido entrenado correctamente.')
    return model


def prediction(model, test_data):
    """

    :param model:
    :param test_data:
    :return:
    """
    test_label = test_data.pop(label_column)
    y_pred = model.predict(test_data)
    logger.info('Se ha hecho una inferencia por lotes exitosamente.')
    return y_pred, test_label


def save_metrics(metrics: dict, path: str):
    with open(path, 'w') as file_path:
        json.dump(metrics, file_path)
    logger.info('Se han guarado las metricas del modelo correctamente')
    return None


def serialize_model(model, model_path: str):
    logger.info('El modelo se ha serializado exitosamente.')
    return joblib.dump(model, filename=model_path)


def main():
    logger.info('El proceso de entrenamiento del modelo ha iniciado ...')
    train_data = load_data(filename='data/train.csv', sep=';',
                           dtype=feature_columns_dtypes.update(label_column_dtype))
    test_data = load_data(filename='data/test.csv', sep=';',
                          dtype=feature_columns_dtypes.update(label_column_dtype))
    preprocessor = define_preprocessing_pipeline()
    model = train(transformer_pipeline=preprocessor, train_data=train_data)
    serialize_model(model=model, model_path='models/gboosting.pkl')
    y_pred, label = prediction(model=model, test_data=test_data)
    metrics = calculate_metrics(y_true=label, y_pred=y_pred)
    save_metrics(metrics, path='metrics/metrics.json')
    metrics_summary(metrics=metrics)
    logger.info('El proceso de entrenamiento del modelo ha concluido exitosamente.')
    return None


if __name__ == '__main__':
    main()
