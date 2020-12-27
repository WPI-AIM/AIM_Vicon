import pandas
import numpy as np
import copy


class Interpolation(object):

    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def interpolate(self):
        raise NotImplementedError


def akmia(sub_value, verbose, category, sub_key, key ):
    s = pandas.Series(sub_value["data"])
    #  Akima interpolation only covers interior NaNs,
    #  and splines are *way* too imprecise with unset boundary conditions,
    #  so linear interpolation is used for unset values at the edges
    try:
        s = s.interpolate(method='akima', limit_direction='both')
    except ValueError:
        if verbose:
            print("Akima Interpolation failed for field " + sub_key + ", in subject " + key + \
                  ", in category " + category + "!")
            print("Falling back to linear interpolation...")
    s = s.interpolate(method='linear', limit_direction='both')
    return s.to_list()


if __name__ == '__main__':

    data = np.array([ 56, 36, np.nan,np.nan,np.nan,np.nan,np.nan,np.nan, 36, np.nan ])
    nan_index = np.argwhere(np.isnan(data))
    print(nan_index.flatten().tolist())
    nums = sorted(set(nan_index.flatten().tolist()))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    index = list(zip(edges, edges))

    print(index)