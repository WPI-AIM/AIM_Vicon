import pandas
import numpy as np
import copy

def akmia(sub_value, verbose, category,sub_key, key ):
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



#
#
def velocity_method(sub_value, verbose, category,sub_key, key):
    data = sub_value["data"]
    new_data = copy.copy(data)
    # get the location of the nans in the list
    nan_index = np.argwhere(np.isnan(data))
    print(nan_index.flatten().tolist())
    nums = sorted(set(nan_index.flatten().tolist()))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    index = list(zip(edges, edges))

    for loc in index:

        diff = abs(loc[0] - loc[1])

        P0 = np.array([ data[loc[0] - 3], data[loc[0] - 2],data[loc[0] - 1] ])





    if verbose:
        print("Akima Interpolation failed for field " + sub_key + ", in subject " + key + \
              ", in category " + category + "!")




if __name__ == '__main__':

    data = np.array([ 56, 36, np.nan,np.nan,np.nan,np.nan,np.nan,np.nan, 36, np.nan ])
    nan_index = np.argwhere(np.isnan(data))
    print(nan_index.flatten().tolist())
    nums = sorted(set(nan_index.flatten().tolist()))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    index = list(zip(edges, edges))

    print(index)