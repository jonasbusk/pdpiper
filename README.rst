Pandas Piper
============

Pipelines for Pandas dataframes.

API inspired by sklearn: http://arxiv.org/abs/1309.0238.

Install::

  $ cd path/to/pdpiper
  $ pip install -U .

Install for development::

  $ cd path/to/pdpiper
  $ pip install -e .

Simple usage example::

  >>> import pandas as pd
  >>> from pdpiper.pipeline import Pipeline
  >>> from pdpiper.transformers import *
  >>> df = pd.DataFrame({'x': [1,1,1,2,2,2], 'y': [1,2,3,1,2,3]})
  >>> pipeline = Pipeline([
  ...     GroupBy(by='x', transformer=Shift(['y'], [1]), drop=True),
  ...     FillNA(columns=['y_shift1'], value=0)
  ... ])
  >>> pipeline(df)
     x  y  y_shift1
  0  1  1       0.0
  1  1  2       1.0
  2  1  3       2.0
  3  2  1       0.0
  4  2  2       1.0
  5  2  3       2.0
