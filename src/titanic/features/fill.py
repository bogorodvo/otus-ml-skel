"""Script Info"""
from typing import List, Text
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


__all__ = ["embarked_imputer", "fill_embarked", "fill_age"]


def embarked_imputer() -> SimpleImputer:
    """Function Info"""
    return SimpleImputer(strategy="most_frequent")


def fill_embarked(data: pd.DataFrame) -> pd.Series:
    """Function Info"""
    return embarked_imputer().fit_transform(data[["Embarked"]])


def fill_age(
    data: pd.DataFrame, group_columns: List, target_column: Text
) -> pd.DataFrame:
    """Function Info"""
    # columns = ['Pclass', 'Sex', 'Embarked', 'is_alone']
    sample = data.loc[:, group_columns + [target_column]].copy()

    for col_name in group_columns:
        sample.loc[:, col_name] = sample.loc[:, col_name].astype(str)

    sample = pd.merge(
        sample.loc[:, group_columns + [target_column]].rename(
            {target_column: f"{target_column}_init"}, axis=1
        ),
        sample.groupby(by=group_columns)
        .aggregate({target_column: "mean"})
        .reset_index(),
        on=group_columns,
        how="left",
    )
    sample.loc[~sample[f"{target_column}_init"].isnull(), target_column] = sample[
        f"{target_column}_init"
    ]
    sample[target_column] = np.floor(
        sample[target_column].replace(np.nan, sample[target_column].mean())
    )
    return sample[target_column]
