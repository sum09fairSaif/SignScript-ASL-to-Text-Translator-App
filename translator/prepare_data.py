# Cleans and splits the raw ASLG-PC12 gloss/English pairs for fine-tuning.
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from translator_config import PREPARED_DATA_DIR, RANDOM_STATE, RAW_DATA_PATH, TEST_SPLIT, VAL_SPLIT


def clean(series):
    # Raw text has a leading BOM (﻿) on some rows and trailing newlines
    return series.str.replace('﻿', '', regex=False).str.strip()


def main():
    if not os.path.exists(RAW_DATA_PATH):
        raise RuntimeError(
            f"Missing {RAW_DATA_PATH} -- download the ASLG-PC12 parquet file first "
            "(https://huggingface.co/datasets/achrafothman/aslg_pc12)"
        )

    df = pd.read_parquet(RAW_DATA_PATH)
    df['gloss'] = clean(df['gloss'])
    df['text'] = clean(df['text'])

    before = len(df)
    df = df[(df['gloss'].str.len() > 0) & (df['text'].str.len() > 0)].drop_duplicates()
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} empty/duplicate row(s)")

    train_df, temp_df = train_test_split(df, test_size=VAL_SPLIT + TEST_SPLIT, random_state=RANDOM_STATE)
    val_fraction = VAL_SPLIT / (VAL_SPLIT + TEST_SPLIT)
    val_df, test_df = train_test_split(temp_df, test_size=1 - val_fraction, random_state=RANDOM_STATE)

    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    os.makedirs(PREPARED_DATA_DIR, exist_ok=True)
    train_df.to_parquet(os.path.join(PREPARED_DATA_DIR, 'train.parquet'))
    val_df.to_parquet(os.path.join(PREPARED_DATA_DIR, 'val.parquet'))
    test_df.to_parquet(os.path.join(PREPARED_DATA_DIR, 'test.parquet'))
    print(f"Saved prepared data to {PREPARED_DATA_DIR}")


if __name__ == '__main__':
    main()
