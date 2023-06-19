"""
Script to test feature functionality
"""
import pytest
import pandas as pd
import numpy as np

from titanic.features.fill import fill_age


@pytest.mark.parametrize(
    "input_size,group_columns,target_column",
    [
        (100, ["A", "B", "C", "D"], "Age"),
        (1, ["A", "B"], "Age"),
        (100000, ["A", "B", "C", "D"], "Age"),
    ],
)
def test_family_age_notnull(input_size, group_columns, target_column):
    """
    If any value in the family age function test is not null,
    then the family age is considered not null.
    """

    data = pd.DataFrame(
        np.random.randint(0, 3, (input_size, len(group_columns)), np.int8),
        columns=group_columns,
    )
    data[target_column] = np.random.randint(0, 100, input_size, np.int8)

    for col_id in range(data.shape[1]):
        data.iloc[np.random.randint(0, 2, input_size, bool), col_id] = np.nan

    if not data[target_column].isnull().all():

        assert not fill_age(data, group_columns, target_column).isnull().any()

        assert np.all(
            np.floor(data[target_column][~data[target_column].isnull()])
            == fill_age(data, group_columns, target_column)[
                ~data[target_column].isnull()
            ]
        )

        assert (
            fill_age(data, group_columns, target_column)[
                ~data[target_column].isnull()
            ].min()
            >= 0
        )

        assert (
            fill_age(data, group_columns, target_column)[
                ~data[target_column].isnull()
            ].max()
            < 100
        )
