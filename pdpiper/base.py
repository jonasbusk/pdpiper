

class BaseStep:
    """Base class for pipeline steps."""

    def fit(self, df):
        return self

    def get_params(self):
        return self.__dict__

    def set_params(self, params):
        self.__dict__ = params


class TransformerMixin:
    """Estimator mixin for transformers."""

    def transform(self, df):
        raise NotImplementedError()

    def fit_transform(self, df):
        return self.fit(df).transform(df)

    def __call__(self, df):
        return self.transform(df)
