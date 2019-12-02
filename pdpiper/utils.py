from multiprocessing import Pool, cpu_count
import pandas as pd


def parallel_apply(grouped, func, concat=True):
    """Parallel equivalent to pandas groupby apply.

    Used to speed up heavy computation on groups of data.

    Example: res = parallel_apply(df.groupby('user_id'), f)

    :param grouped: Pandas groupby object.
    :param func: Function to apply to each group.
    :param concat: Whether to concatenate the result with pandas.
    :return: DataFrame or list of results.
    """
    with Pool(cpu_count()) as p:
        result_list = p.map(func, [group for name, group in grouped])
    if concat:
        return pd.concat(result_list)
    else:
        return result_list
