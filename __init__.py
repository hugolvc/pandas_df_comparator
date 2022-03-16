# from locale import dcgettext
import json
import numpy as np
import pandas as pd
from typing import Union, Dict, List, Any
from pydoc import locate


class Comparator:
    def __init__(
        self,
        df1: pd.DataFrame, df1_name: str,
        df2: pd.DataFrame, df2_name: str,
            adjustments: Union[Dict[str, str], None] = None):

        self.df1 = df1.copy()
        self.df1_name = df1_name

        self.df2 = df2.copy()
        self.df2_name = df2_name

        self.adjustments = {}
        self.set_adjustments(adjustments)

    def save_adjustments(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.adjustments, f, indent=4)

    def load_adjustments(self, path: str):
        with open(path, 'r') as f:
            adjustments = json.load(f)

        self.set_adjustments(adjustments)

    def show_adustments(self):
        print(json.dumps(self.adjustments, indent=4))

    def set_adjustments(
        self,
        adjustments: Dict[str, Dict[str, Union[
            str, Dict[str, int], Dict[str, str]]]]):

        if adjustments:
            if 'df1' in adjustments.keys():
                if 'df1' not in self.adjustments:
                    self.adjustments['df1'] = {}

                for column, column_adjustments in adjustments['df1'].items():
                    if column not in self.adjustments['df1']:
                        self.adjustments['df1'][column] = []

                    self.adjustments['df1'][column].append(column_adjustments)

                    self.execute_column_adjustments(
                        'df1', column, column_adjustments)

            if 'df2' in adjustments.keys():
                if 'df2' not in self.adjustments:
                    self.adjustments['df2'] = {}

                for column, column_adjustments in adjustments['df2'].items():
                    if column not in self.adjustments['df2']:
                        self.adjustments['df2'][column] = []

                    self.adjustments['df2'][column].append(column_adjustments)

                    self.execute_column_adjustments(
                        'df2', column, column_adjustments)

    def execute_column_adjustments(
        self, df: str, column: str,
            adjustments: Dict[str, Union[
                str, Dict[str, int], Dict[str, str]]]):

        adjustment_cases = {
            'astype': self.adjust_type,
            'zfill': self.adjust_zfill,
            'lower': self.adjust_lower,
            'upper': self.adjust_upper,
            'replace': self.adjust_replace,
            'str_replace': self.adjust_str_replace,
            'make_int': self.adjust_make_int,
            'fillna': self.adjust_fillna,
            'round': self.adjust_round,
        }

        for adjustment in adjustments:
            if adjustment.__class__.__name__ == 'str':
                adjustment_case = adjustment
                adjustment_cases[adjustment_case](df, column)
            else:
                adjustment_case = list(adjustment.keys())[0]
                adjustment_cases[adjustment_case](
                    df, column, adjustment[list(adjustment.keys())[0]])

    def adjust_round(self, df_str: str, column: str, digits: int):

        df = self.df1 if df_str == 'df1' else self.df2

        df[column] = df[column].round(digits)

    def adjust_make_int(
        self, df_str: str, column: str,
            values: Dict[str, Dict[str, Any]]):
        df = self.df1 if df_str == 'df1' else self.df2

        df[column] = df[column].replace({np.Inf: values['inf']})
        df[column] = df[column].fillna(values['nan'])
        df[column] = df[column].astype({column: int})

    def adjust_fillna(self, df: str, column: str, value: Any):
        if df == 'df1':
            self.df1[column] = self.df1[column].fillna(value)
        else:
            self.df2[column] = self.df2[column].fillna(value)

    def adjust_str_replace(
            self, df: str, column: str, replace: List[str]):

        if df == 'df1':
            self.df1[column] =\
                self.df1[column].str.replace(replace[0], replace[1])
        else:
            self.df2[column] =\
                self.df2[column].str.replace(replace[0], replace[1])

    def adjust_replace(
            self, df: str, column: str, replace: Dict[Any, Any]):

        if df == 'df1':
            self.df1[column] == self.df1[column].replace(replace)
        else:
            self.df2[column] == self.df2[column].replace(replace)

    def adjust_upper(self, df: str, column: str):
        if df == 'df1':
            self.df1[column] = self.df1[column].str.upper()
        else:
            self.df2[column] = self.df2[column].str.upper()

    def adjust_lower(self, df: str, column: str):
        if df == 'df1':
            self.df1[column] = self.df1[column].str.lower()
        else:
            self.df2[column] = self.df2[column].str.lower()

    def adjust_type(self, df: str, column: str, new_type: 'str'):
        if df == 'df1':
            self.df1 = self.df1.astype({column: locate(new_type)})
        else:
            self.df2 = self.df2.astype({column: locate(new_type)})

    def adjust_zfill(self, df: str, column: str, adjustment: int):
        if df == 'df1':
            self.df1[column] =\
                self.df1[column].str.zfill(adjustment)
        else:
            self.df2[column] =\
                self.df2[column].str.zfill(adjustment)

    def compare_shapes(self):
        df1_columns = list(self.df1.columns)
        df2_columns = list(self.df2.columns)

        cols = set(df1_columns + df2_columns)
        comp_cols = []
        for col in cols:
            comp_cols.append([
                col,
                '*' if col in df1_columns else '',
                '*' if col in df2_columns else ''])
        comp_cols.append(['size', len(self.df1), len(self.df2)])

        result = pd.DataFrame(
            comp_cols, columns=['col', self.df1_name, self.df2_name])

        print(result.to_markdown())

    def compare_sizes(self, columns: List[str] = []):
        """Compares the incidences of every data value for every column of two
        dataframes. It prints on stdout, for every value detected in each
        dataframe column, the amount of incidences on each one, and a column
        that highlights those values where both dataframes don't match.

        Args:
            dataframe1 (pd.DataFrame): First dtaframe to compare.
            df1_name (str): The name of the first dataframe to print out.
            dataframe2 (pd.DataFrame): Second dataframe to compare
            df2_name (str): The name of the second dataframe to print out.
            columns (List[str], optional, default: []): a list of
                columns to consider.
                Columns prefixed with a hyphen are avoided. If not provided
                or empty, all columns will be considered. If the list contains
                columns to consider and columns to avoid, the latest are
                ignored.
        """

        avoid =\
            [column[1:] for column in columns if column.startswith('-')]

        columns =\
            [column[1:] if column.startswith('-') else column
             for column in columns]

        columns = [column for column in columns if column not in avoid]

        if len(columns) == 0:
            df1_columns =\
                [column for column in self.df1.columns if column not in avoid]
            df2_columns =\
                [column for column in self.df2.columns if column not in avoid]
        else:
            df1_columns =\
                [column for column in self.df1.columns if column in columns]
            df2_columns = [column for column in self.df2.columns
                           if column in columns]
        columns = set(df1_columns + df2_columns)

        columns_in_df1 =\
            [column for column in df1_columns if column not in df2_columns]
        df1_sizes = pd.DataFrame()
        for column in columns_in_df1:
            if column in columns:
                df1_sizes = pd.DataFrame(self.df1.groupby(column).size())
                df1_sizes.rename(columns={0: self.df1_name}, inplace=True)
                df1_sizes[self.df2_name] = 0
                df1_sizes['error'] = '*'
                df1_sizes = df1_sizes.replace({np.nan: 0})
                print('\n' + df1_sizes.to_markdown())
                print("mismatches: " + str(len(df1_sizes)))

        columns_in_df1_and_df2 =\
            [column for column in df1_columns if column in df2_columns]
        df1_and_df2_sizes = pd.DataFrame()
        for column in columns_in_df1_and_df2:
            if column in columns:
                df1_df2_sizes =\
                    pd.DataFrame(self.df1.groupby(
                        column, as_index=False).size())
                df1_df2_sizes.rename(
                    columns={'size': self.df1_name}, inplace=True)
                df1_df2_sizes = df1_df2_sizes.astype({self.df1_name: str})

                df2_df1_sizes =\
                    pd.DataFrame(self.df2.groupby(
                        column, as_index=False).size())
                df2_df1_sizes.rename(
                    columns={'size': self.df2_name}, inplace=True)
                df2_df1_sizes = df2_df1_sizes.astype({self.df2_name: str})

                df1_and_df2_sizes = df1_df2_sizes.\
                    merge(df2_df1_sizes, on=column, how='outer')

                df1_and_df2_sizes['error'] = np.where(
                    df1_and_df2_sizes[self.df1_name]
                    != df1_and_df2_sizes[self.df2_name],
                    '*', '')
                df1_and_df2_sizes = df1_and_df2_sizes.replace({np.nan: 0})
                df1_and_df2_sizes.reset_index(inplace=True, drop=True)
                df1_and_df2_sizes = df1_and_df2_sizes.append([
                    {
                        column: 'dtype',
                        self.df1_name: self.df1[column].dtype,
                        self.df2_name: self.df2[column].dtype,
                        'error': '*' if self.df1[column].dtype
                        != self.df2[column].dtype else ''
                    }
                ], ignore_index=True)
                print('\n' + df1_and_df2_sizes.to_markdown())
                print("mismatches: " + str(len(
                    df1_and_df2_sizes.loc[df1_and_df2_sizes['error'] == '*'])))

        columns_in_df2 =\
            [column for column in df2_columns if column not in df1_columns]
        df2_sizes = pd.DataFrame()
        for column in columns_in_df2:
            if column not in avoid:
                df2_sizes = pd.DataFrame(self.df2.groupby(column).size())
                df2_sizes.rename(columns={0: self.df2_name}, inplace=True)
                df2_sizes[self.df1_name] = 0
                df2_sizes['error'] = '*'
                df2_sizes = df2_sizes.replace({np.nan: 0})
                print('\n' + df2_sizes.to_markdown())
                print("mismatches: " + str(len(df2_sizes)))
