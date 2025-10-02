import pandas as pd
import os
from predictor import Predictor
from utils.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR

def main():

    predictor = Predictor()

    df = pd.read_csv(os.path.join(RAW_DATA_DIR, "test.csv"), delimiter=";")
    results = []
    for text in df['sample']:
        results.append(predictor.predict(text=text))

    df['annotation'] = results
    df.to_csv(os.path.join(PROCESSED_DATA_DIR, "submission.csv"), sep=";", index=False)


if __name__ == "__main__":
    main()