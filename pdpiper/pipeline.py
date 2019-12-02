from .base import BaseStep, TransformerMixin


class Pipeline(BaseStep, TransformerMixin):
    """Pipeline of steps."""

    def __init__(self, steps=[]):
        self.steps = steps

    def fit(self, df):
        """Fit each step in sequence."""
        self.fit_transform(df)
        return self

    def get_params(self):
        """Get params of each step as list."""
        return [s.get_params() for s in self.steps]

    def set_params(self, params_list):
        """Set params of each step from list of params."""
        assert len(self.steps) == len(params_list)
        for step, params in zip(self.steps, params_list):
            step.set_params(params)

    def transform(self, df):
        """Apply each step in sequence."""
        for step in self.steps:
            df = step.transform(df)
        return df

    def fit_transform(self, df):
        """Fit and apply each step in sequence."""
        for step in self.steps:
            df = step.fit_transform(df)
        return df
