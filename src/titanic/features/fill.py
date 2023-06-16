import pandas
import numpy as np
from sklearn.impute import SimpleImputer


__all__ = ["embarked_imputer", "fill_embarked", "fill_age"]


def embarked_imputer() -> SimpleImputer:
    return SimpleImputer(strategy="most_frequent")


def fill_embarked(df: pandas.DataFrame) -> pandas.Series:
    return embarked_imputer().fit_transform(df[["Embarked"]])
    
    
def fill_age(data: pandas.DataFrame) -> pandas.DataFrame:
    
    columns = ['Pclass','Sex','Embarked','is_alone']
    return pandas.Series(
        pandas.merge(data.loc[:, columns],
                 data.groupby(columns)['Age'].count().reset_index(), 
                 on=columns, how='left')['Age'], 
        index=data.index).replace(np.nan, data.loc[:, 'Age'].mean())

