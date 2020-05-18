import os
import sys
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from utilities.config import input_filename, label_column, \
                             feature_columns_dtypes, label_column_dtype

logger = logging.getLogger('split_data')
logger.setLevel(logging.INFO)
console_handle = logging.StreamHandler(sys.stdout)
console_handle.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s -%(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console_handle.setFormatter(formatter)
logger.addHandler(console_handle)


def load_data(filename: str, **kwargs) -> pd.DataFrame:
    """

    :param filename:
    :return: pd.DataFrame
    """
    df = pd.read_csv(filename, **kwargs)
    logger.info('Se han cargado los datos correctamente.')
    return df


def split_data(df: pd.DataFrame, label: str, stratify: bool = False):
    """

    :param stratify:
    :param df:
    :param label:
    :return:
    """
    target = df.pop(label)
    if stratify:
        train_data, test_data, train_label, test_label = train_test_split(df, target, test_size=0.3,
                                                                          random_state=42, stratify=target)
    else:
        train_data, test_data, train_label, test_label = train_test_split(df, target, test_size=0.3, random_state=42)
    logger.info('Se ha partido el conjunto de datos en conjuntos de entrenamiento, validación y prueba.')
    return train_data, test_data, train_label, test_label


def concatenate_data(first_df: pd.DataFrame, second_df: pd.DataFrame) -> pd.DataFrame:
    """

    :param first_df:
    :param second_df:
    :return:
    """
    df = pd.concat([first_df, second_df], axis=1).reset_index(drop=True)
    logger.info('Se han unido ambos set de datos.')
    return df


def export_data(df: pd.DataFrame, path: str, with_header: bool = False):
    """

    :param df:
    :param path:
    :param with_header:
    :return:
    """
    logger.info('El archivo se ha exportado como un csv satisfactoriamente.')
    if with_header:
        return df.to_csv(path, sep=';', index=False, header=True)
    else:
        return df.to_csv(path, sep=';', index=False, header=False)


def delete_file(filename: str):
    """

    :param filename:
    :return:
    """
    if os.path.exists(filename):
        os.remove(filename)
        logger.info('El archivo se ha eliminado correctamente.')
    else:
        logger.info('El archivo no existe.')


def main():
    df = load_data(filename=f'data/{input_filename}', dtype=feature_columns_dtypes.update(label_column_dtype))
    train_data, test_data, train_label, test_label = split_data(df=df,
                                                                label=label_column,
                                                                stratify=True)
    train_df = concatenate_data(train_data, train_label)
    test_df = concatenate_data(test_data, test_label)
    export_data(df=train_df, path='data/train.csv', with_header=True)
    export_data(df=test_df, path='data/test.csv', with_header=True)
    logger.info('El proceso ha finalizado exitosamente')
    return None


if __name__ == '__main__':
    main()
