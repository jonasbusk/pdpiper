import numpy as np
import pandas as pd

from .base import BaseStep, TransformerMixin
from .utils import parallel_apply


# utility

class GroupBy(BaseStep, TransformerMixin):
    """Group by column(s) and transform."""

    def __init__(self, by, transformer=lambda df: df, drop=False):
        self.by = by
        self.transformer = transformer
        self.drop = drop

    def transform(self, df):
         return df.groupby(self.by).apply(self.transformer).reset_index(drop=self.drop)


class GroupByParallel(BaseStep, TransformerMixin):
    """Group by column(s) and transform in parallel."""

    def __init__(self, by, transformer=lambda df: df, drop=False):
        self.by = by
        self.transformer = transformer
        self.drop = drop

    def transform(self, df):
         return parallel_apply(df.groupby(self.by), self.transformer).reset_index(drop=self.drop)


class Assert(BaseStep, TransformerMixin):
    """Make assertion on data frame."""

    def __init__(self, condition, msg=''):
        # condition is a function of df
        self.condition = condition
        self.msg = msg

    def transform(self, df):
        assert self.condition(df), self.msg
        return df


# basic

class CopyColumn(BaseStep, TransformerMixin):
    """Copy a column."""

    def __init__(self, column, new_column):
        self.column = column
        self.new_column = new_column

    def transform(self, df):
        df[self.new_column] = df[self.column]
        return df


class AsType(BaseStep, TransformerMixin):
    """Set column types."""

    def __init__(self, column_types):
        self.column_types = column_types

    def transform(self, df):
        for column, typ in self.column_types.iteritems():
            df[column] = df[column].astype(typ)
        return df


class BinarizeNumber(BaseStep, TransformerMixin):
    """Binarize number column and respect nan."""

    def __init__(self, column, inplace=False):
        self.column = column
        self.inplace = inplace

    def transform(self, df):
        column = self.column if self.inplace else self.column+'_bin'
        df.loc[df[self.column].notna() & (df[self.column]==0), column] = 0
        df.loc[df[self.column].notna() & (df[self.column]!=0), column] = 1
        return df


class CreateDummy(BaseStep, TransformerMixin):
    """Create dummy variable from another column."""

    def __init__(self, column, values, postfix=None):
        self.column = column
        self.values = values
        self.new_column = '{}_{}'.format(column, postfix or str(values[0]))

    def transform(self, df):
        df.loc[df[self.column].notna() & ~df[self.column].isin(self.values), self.new_column] = 0
        df.loc[df[self.column].notna() & df[self.column].isin(self.values), self.new_column] = 1
        return df


class DropNA(BaseStep, TransformerMixin):
    """Remove missing data."""

    def __init__(self, axis=0, how='any', thresh=None, subset=None, inplace=False):
        self.axis = axis
        self.how = how
        self.thresh = thresh
        self.subset = subset
        self.inplace = inplace

    def transform(self, df):
        return df.dropna(axis=self.axis, how=self.how, thresh=self.thresh, subset=self.subset,
                         inplace=self.inplace)


class DropDuplicates(BaseStep, TransformerMixin):
    """Drop duplicate rows."""

    def __init__(self, subset=None, keep='first'):
        self.subset = subset
        self.keep = keep

    def transform(self, df):
        return df.drop_duplicates(subset=self.subset, keep=self.keep)


class FilterColumns(BaseStep, TransformerMixin):
    """Filter columns."""

    def __init__(self, columns, dropna=False):
        self.columns = columns
        self.dropna = dropna

    def transform(self, df):
        return df[self.columns].dropna() if self.dropna else df[self.columns]


class CountNA(BaseStep, TransformerMixin):
    """Count missing values in rows."""

    def __init__(self, columns=None, name='nans'):
        self.columns = columns
        self.name = name

    def transform(self, df):
        columns = self.columns or df.columns
        df[self.name] = df[columns].isnull().sum(axis=1)
        return df


class FillNA(BaseStep, TransformerMixin):
    """Fill missing values with pandas fillna method."""

    def __init__(self, columns, value=None, method=None, axis=None, inplace=False, limit=None):
        self.columns = columns
        self.value = value
        self.method = method
        self.axis = axis
        self.inplace = inplace
        self.limit = limit

    def transform(self, df):
        if not self.inplace:
            df = df.copy()
        for c in self.columns:
            df[c] = df[c].fillna(value=self.value, method=self.method, axis=self.axis,
                                 limit=self.limit)
        return df


class Interpolate(BaseStep, TransformerMixin):
    """Fill missing values with pandas interpolate method."""

    def __init__(self, columns, method='linear', axis=0, limit=None, inplace=False,
                 limit_direction='forward', limit_area=None):
        self.columns = columns
        self.method = method
        self.axis = axis
        self.limit = limit
        self.inplace = inplace
        self.limit_direction = limit_direction
        self.limit_area = limit_area

    def transform(self, df):
        if not self.inplace:
            df = df.copy()
        for c in self.columns:
            df[c] = df[c].interpolate(method=self.method, axis=self.axis, limit=self.limit,
                               limit_direction=self.limit_direction, limit_area=self.limit_area)
        return df


class IsIn(BaseStep, TransformerMixin):
    "Filter rows with isin."

    def __init__(self, column, values):
        self.column = column
        self.values = values

    def transform(self, df):
        return df.loc[df[self.column].isin(self.values)]


class MinRowsInGroup(BaseStep, TransformerMixin):
    """Group by column and ensure min number of rows in each group."""

    def __init__(self, column, min_rows):
        self.column = column
        self.min_rows = min_rows

    def transform(self, df):
        return df[df.groupby(self.column)[self.column].transform(len) >= self.min_rows]


class MinValsInGroup(BaseStep, TransformerMixin):
    """Group by column and ensure min number of unique vals in each group."""

    def __init__(self, groupby, column, min_vals):
        self.groupby = groupby
        self.column = column
        self.min_vals = min_vals

    def transform(self, df):
        return df[df.groupby(self.groupby)[self.column].transform(lambda x: len(np.unique(x))) >= self.min_vals]


class RangeScale(BaseStep, TransformerMixin):
    """Scale column to range."""

    def __init__(self, column, to_range=[0,1], from_range=None, inplace=False, ignore=False):
        self.column = column
        self.to_range = to_range
        self.from_range = from_range
        self.inplace = inplace
        self.ignore = ignore

    def transform(self, df):
        if self.column not in df.columns and self.ignore:
            return df
        if self.from_range is None:
            self.from_range = [df[self.column].min(), df[self.column].max()]
        c = df[self.column]
        a, b = self.from_range[0], self.from_range[1]
        i, j = self.to_range[0], self.to_range[1]
        scaled = (c-a)*(j-i)/(b-a)+i
        column = self.column if self.inplace else self.column+'_scaled'
        df[column] = scaled
        return df


class Log(BaseStep, TransformerMixin):
    """Logarithmic transform."""

    def __init__(self, column, offset=0, inplace=False):
        self.column = column
        self.offset = offset
        self.inplace = inplace

    def transform(self, df):
        column = self.column if self.inplace else self.column+'_log'
        df[column] = (df[self.column] + self.offset).apply(np.log)
        return df


class Exp(BaseStep, TransformerMixin):
    """Exponential transform."""

    def __init__(self, column, inplace=False):
        self.column = column
        self.inplace = inplace

    def transform(self, df):
        column = self.column if self.inplace else self.column+'_exp'
        df[column] = df[self.column].apply(np.exp)
        return df


class SubtractMean(BaseStep, TransformerMixin):
    """Subtract mean."""

    def __init__(self, column, inplace=False):
        self.column = column
        self.inplace = inplace

    def transform(self, df):
        column = self.column if self.inplace else self.column+'_submean'
        df[column] = df[self.column] - df[self.column].mean()
        return df


class CreateNumber(BaseStep, TransformerMixin):
    """Sort by column and add row number."""

    def __init__(self, column='date', new_column='number', start=1, repeat=1):
        self.column = column
        self.new_column = new_column
        self.start = start
        self.repeat = repeat

    def transform(self, df):
        df = df.sort_values(by=self.column)
        df[self.new_column] = (np.arange(self.start, df.shape[0]+self.start)-self.start) / \
                                                                         self.repeat + self.start
        return df


class Maximum(BaseStep, TransformerMixin):

    def __init__(self, column, new_column, value=0):
        self.column = column
        self.new_column = new_column
        self.value = value

    def transform(self, df):
        with np.errstate(invalid='ignore'):
            df[self.new_column] = np.maximum(df[self.column], self.value)
        return df


class Minimum(BaseStep, TransformerMixin):

    def __init__(self, column, new_column, value=0):
        self.column = column
        self.new_column = new_column
        self.value = value

    def transform(self, df):
        with np.errstate(invalid='ignore'):
            df[self.new_column] = np.minimum(df[self.column], self.value)
        return df


# timeseries

class CreateDate(BaseStep, TransformerMixin):
    """Create date column from timestamp column"""

    def __init__(self, column='timestamp', new_column='date', unit='ns'):
        self.column = column
        self.new_column = new_column
        self.unit = unit

    def transform(self, df):
        df[self.new_column] = pd.to_datetime(df[self.column],
                                             unit=self.unit).dt.date.astype('datetime64[ns]')
        return df


class CreateDatetime(BaseStep, TransformerMixin):
    """Create datetime column from timestamp column"""

    def __init__(self, column='timestamp', new_column='datetime', unit='ns'):
        self.column = column
        self.new_column = new_column
        self.unit = unit

    def transform(self, df):
        df[self.new_column] = pd.to_datetime(df[self.column],
                                             unit=self.unit).astype('datetime64[ns]')
        return df


class FillMissingDates(BaseStep, TransformerMixin):
    """Fill missing dates in df by creating a new row for each missing date."""

    def __init__(self, start=None, end=None, column='date'):
        self.start = start
        self.end = end
        self.column = column

    def transform(self, df):
        if not df.empty:
            start = self.start or df[self.column].min()
            end = self.end or df[self.column].max()
            idx = pd.date_range(start, end) # create complete date series from start to end
            df = df.sort_values(by=self.column)
            df = df.set_index(self.column)
            df = df.reindex(idx) # reindex by complete date series
            df = df.reset_index()
            df = df.rename(columns={'index': self.column})
        return df


class DayOfWeek(BaseStep, TransformerMixin):
    """Compute day of week from dates."""

    def __init__(self, column='date', new_column='day_of_week', one_hot=False):
        self.column = column
        self.new_column = new_column
        self.one_hot = one_hot

    def transform(self, df):
        if self.one_hot:
            df1 = pd.get_dummies(df[self.column].dt.dayofweek, prefix=self.new_column)
            df = pd.concat([df, df1], axis=1)
        else:
            df[self.new_column] = df[self.column].dt.dayofweek
        return df


class Month(BaseStep, TransformerMixin):
    """Compute month of year from dates."""

    def __init__(self, column='date', new_column='month', one_hot=False):
        self.column = column
        self.new_column = new_column
        self.one_hot = one_hot

    def transform(self, df):
        if self.one_hot:
            df1 = pd.get_dummies(df[self.column].dt.month, prefix=self.new_column)
            df = pd.concat([df, df1], axis=1)
        else:
            df[self.new_column] = df[self.column].dt.month
        return df


class Shift(BaseStep, TransformerMixin):
    """Create shifted features."""

    def __init__(self, columns, shifts=[1]):
        self.columns = columns
        self.shifts = shifts

    def transform(self, df):
        for col in self.columns:
            for i in self.shifts:
                df['{:s}_shift{:d}'.format(col, i)] = df[col].shift(i)
        return df


class ConsecutiveDifference(BaseStep, TransformerMixin):
    """Compute difference of consecutive rows and shift.

    Examples:
        shift=1 is the difference between today and yesterday
        shift=2 is the difference between yesterday and the day before
    """

    def __init__(self, columns, shifts=[1]):
        self.columns = columns
        self.shifts = shifts

    def transform(self, df):
        for col in self.columns:
            diff = df[col] - df[col].shift() # consecutive diff column
            for i in self.shifts:
                df['{:s}_condiff{:d}'.format(col, i)] = diff.shift(i-1) # add shifted diff columns
        return df


class CumulativeDifference(BaseStep, TransformerMixin):
    """Compute cumulative difference of shifted columns.

    Examples:
        shift=1 is the difference between today and yesterday
        shift=2 is the difference between today and 2 days ago
    """

    def __init__(self, columns, shifts=[1]):
        self.columns = columns
        self.shifts = shifts

    def transform(self, df):
        for col in self.columns:
            for i in self.shifts:
                df['{:s}_cumdiff{:d}'.format(col, i)] = df[col] - df[col].shift(i)
        return df


class RollingMeanDifference(BaseStep, TransformerMixin):
    """Compute difference from moving average features.

    Examples:
      shift=1, window=2 is the difference between today and the mean of yesterday and 2 days ago
      shift=2, window=2 is the difference between yesterday and the mean of 2 days and 3 days ago
    """

    def __init__(self, columns, shifts=[1], window=2):
        self.columns = columns
        self.shifts = shifts
        self.window = window

    def transform(self, df):
        for col in self.columns:
            diff = df[col] - df[col].rolling(window=self.window).mean().shift() # compute diff from moving avg
            for i in self.shifts:
                df['{:s}_rmdiff{:d}'.format(col, i)] = diff.shift(i-1)
        return df
