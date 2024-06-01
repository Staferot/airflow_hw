import json
import logging
import os
from pathlib import Path

import dill
import pandas as pd

MODEL_VERSION = '1.0.0'
PATH = os.environ.get('PROJECT_PATH', '.')
MODEL_DIR_PATH = f'{PATH}/data/models'
TEST_DIR_PATH = f'{PATH}/data/test'
PREDICTION_DIR_PATH = f'{PATH}/data/predictions'


def load_model():
    with open(f'{MODEL_DIR_PATH}/cars_pipe_{MODEL_VERSION}.pkl', 'rb') as f:
        return dill.load(f)


def load_tests():
    test_dir = Path(TEST_DIR_PATH)
    test_json = [json.load(el.open('rb')) for el in test_dir.iterdir() if el.is_file() and el.name.endswith('.json')]
    return pd.DataFrame.from_records(test_json)


def save_predictions(df):
    df.to_csv(f'{PREDICTION_DIR_PATH}/test_{MODEL_VERSION}.csv', index=False)


def predict():
    logging.info('Loading model...')
    predicted_model = load_model()

    logging.info('Loading tests...')
    test_data = load_tests()

    logging.info('Predicting...')
    test_data['price_category'] = predicted_model.predict(test_data)

    logging.info('Saving predictions...')
    save_predictions(test_data)

    logging.info('Done!')


if __name__ == '__main__':
    predict()
