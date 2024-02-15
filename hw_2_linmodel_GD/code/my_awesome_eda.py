import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def count_outliers(x)->int:
    '''
    Calculate the number of outliers based on the rule: an outlier is a point that is located further than 1.5*IQR.
    
    arguments:
        - x (pd.series): pandas series (column) for calculating number of outliers in it
    return:
        - int: number of outliers
    '''
    iqr = np.percentile(x, 75) - np.percentile(x, 25)
    n_outliers = np.sum((x < (np.percentile(x, 25) - 1.5 * iqr)) | (x > (np.percentile(x, 75) + 1.5 * iqr)))
    return n_outliers 



def run_eda(df):
    '''
    Provide Exploratory Data Analysis.
    - The number of observations (rows) and variables (columns)
    - Data types in columns (numeric, string, factor or other (including date format))
    - Visualization of the percentage of missing values for each column
    - Visualization of heatmap correlations across all variables
    - Visualization of boxplots for each numerical variable
    
    arguments:
        - df (pd.dataframe): pandas dataframe
    '''
    print("This dataframe has", df.shape[0], 'samples (rows) and', df.shape[1], 'columns (variables)\n')
    numeric_columns = []
    string_columns = []
    factor_columns = []
    other_columns = []
    types = df.dtypes.to_dict()
    for column, ctype in types.items():
        if column.find('ID')!= -1 or column.find('Id')!= -1:
            string_columns.append(column)
        elif (df[column].dropna()).nunique() / len(df[column].dropna()) < 0.1:
            factor_columns.append(column)
        elif (ctype == 'int64') or (ctype == 'float64'):
            numeric_columns.append(column)
        elif ctype == 'bool':
            factor_columns.append(column)
        elif ctype == 'object':
            if (df[column].dropna()).nunique() / len(df[column].dropna()) > 0.7:
                string_columns.append(column)
            else:
                factor_columns.append(column)
        else:
            other_columns.append(column)
    print('This dataframe contains:')
    if len(numeric_columns) != 0:
        print(len(numeric_columns), "numeric variables:", ', '.join(numeric_columns))
    if len(string_columns) != 0:
        print(len(string_columns), "string variables:", ', '.join(string_columns))
    if len(factor_columns) != 0:
        print(len(factor_columns), "factor variables:", ', '.join(factor_columns))
    if len(other_columns) != 0:
        print(len(other_columns), "other variables:", ', '.join(other_columns))
    print('\nStatistic for factor variables:')
    for factor_column in factor_columns:
        print(f"\n---- {factor_column} ---")
        describe_table_for_factor_counts = df[factor_column].value_counts().rename_axis('value').reset_index(name='count')
        describe_table_for_factor_frequencies = df[factor_column].value_counts(normalize = True).rename_axis('value').reset_index(name='frequency')
        describe_table_for_factor = pd.merge(describe_table_for_factor_counts,
                                        describe_table_for_factor_frequencies,
                                        on = 'value')
        display(describe_table_for_factor)
    print('\nStatistic for number variables:')
    display(df[numeric_columns].describe())
    print()
    na_number = df.isna().sum()
    if na_number.sum() == 0:
        print('\nThere are no NA values in this dataframe.')
    elif na_number.sum() == 1:
        print('\nThere is one NA value in this dataframe.\nIn column:', (na_number[na_number>0]).index[0])
    else:
        print('\nThere are', na_number.sum(), 'NA values in total\nThey are in', df.isnull().any(axis=1).sum(), 'rows in this dataframe.\nIn columns:', ', '.join((na_number[na_number>0]).index.tolist()))
    duplicated_number = df.duplicated().sum()
    if duplicated_number == 0:
        print('\nThere are no duplicated rows in this dataframe.')
    elif duplicated_number == 1:
        print('\nThere is one duplicated row in this dataframe.')
    else:
        print('\nThere are', duplicated_number, 'duplicated rows in this dataframe.')
    
    
    fraction_of_na = df.isnull().sum(axis = 0) / len(df)
    clrs = ['#286619' if (x < 0.5) else '#9B075C' for x in fraction_of_na]
    sns.barplot(x = fraction_of_na.index, y = fraction_of_na, palette = clrs)
    plt.ylabel('Fraction of missing values')
    plt.xlabel('Variables')
    plt.title('Missing data')
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.show()

    correlations = df.corr(method='spearman') 
    plt.figure(figsize = (10,10))
    sns.heatmap(correlations, square = True, annot = True, linewidths = 0.25)
    plt.title("Correlation matrix for features", size = 20)
    plt.show()

    numeric = list(df.select_dtypes(include = np.number).columns)
    cols_for_plots = 3
    if len(numeric) % cols_for_plots == 0:
        rows_for_plots = len(numeric) // 3
    else:
        rows_for_plots = len(numeric) // 3 + 1
    fig, axes = plt.subplots(nrows = rows_for_plots, ncols = cols_for_plots, figsize=(26, 14))
    axes = axes.ravel()
    for col, ax in zip(numeric, axes):
        sns.boxplot(y = df[col], ax = ax)
        ax.set_title(col, size = 30)
    plt.show()
    
