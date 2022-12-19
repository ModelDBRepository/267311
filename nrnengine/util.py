# Copyright (c) 2022 Adrian Negrean
# negreanadrian@gmail.com
#
# Software released under MIT license, see license.txt for conditions

import numpy as np, itertools
from collections.abc import Iterable
from copy import deepcopy

flatten_list = lambda l: [item for sublist in l for item in sublist]

def del_list_idxs(l, idxs):
    """
    Deletes one or more elements for a list given their current list indices.

    Parameters
    ----------
    l : list
    idxs : list
        Current indices of elements to delete.

    Return
    ------
    list
    """
    return [i for j, i in enumerate(l) if j not in idxs]

def set_default_keys(src, trgt):
    """
    Helper function to set key values in a target dict if keys don't exist using
    values from a source dict.

    Parameters
    ----------
    src, trgt : dict
        Source and target dict.

    Returns
    -------
    None
    """
    for src_key, src_val in src.items():
        if src_key not in trgt:
            trgt[src_key] = src_val

def set_attr_from_str(obj, attr_str, val, path_marker = '.'):
    """
    Sets an object attribute given a string of the form '<attr1><path_marker><attr2><path_marker> ...'
    
    Parameters
    ----------
    obj : object
        Object to which the value will be assigned

    attr_str : str
        Attribute of the form '<attr1><path_marker><attr2><path_marker> ...'

    val : object
        Value to assign to last object attribute.

    Returns
    -------
    None
        Assigns given value to the last attribute of the object.

    """
    attrs = attr_str.split(path_marker)
    o = obj
    for attr in attrs[:-1]:
        o = getattr(o, attr)
    setattr(o, attrs[-1], val)

def set_dpath(source, target, path_mrkr = '/'):
    """
    Sets target nested dictionary keys to the values in a source dict that has keys coded as a path in the nested target dictionary.
    If path keys described in the source dict do not exist in target dict, then target dict keys are created. If path marker is contained
    in the dict key, when specifying the path name, it must be doubled, e.g. 'key1//key2' for setting a key 'key1/key2'.

    Parameters
    ----------
    source : dict
        Single level dictionary with keys describing a path in the target dictionary using a path marker.

    target : dict
        Nested dictionaries.

    path_mrkr : str
        Path marker.

    Returns
    -------
    None
    """
    for src_key, src_val in source.items():
        path = parse_dict_paths(src_key, path_mrkr)
        # visit nested dict items
        item = target
        for trgt_key in path[:-1]:
            # if dict key exists, then assign value
            if trgt_key in item:
                item = item[trgt_key]
            else:
            # if key does not exist, create key and assign value
                item[trgt_key] = {}
                item = item[trgt_key]
        item[path[-1]] = src_val
        
def parse_dict_paths(path, path_mrkr = '/'):
    """
    Splits a nested dict key path into individual keys. If path_mrkr is contained by a key, it must be doubled, e.g. '//'

    Parameters
    ----------
    path : str
        Nested key path e.g. /key1/key2_1//key2_2 corresponding to a dict of the form {'key1': {'key2_1/key2_2': val}}
    path_mrkr : str
        Path marker, commonly '/'.

    Returns
    -------
    list of str
        Nested dict keys e.g. ['key1', 'key2_1/key2_2']
    """
    out = path.strip(path_mrkr).split(path_mrkr)
    idx = 0
    while idx < len(out):
        if not out[idx]:
            out[idx] = out[idx-1] + path_mrkr + out[idx+1]
            del out[idx-1]
            idx -= 1
            del out[idx+1]
        idx += 1
    return out
    
def dict_product(d):
    """
    Obtains the cartesian product of a dict {key1: val1, key2: val2 ...}. If val1..n is iterable and not a dict, iteration over values is desired, otherwise
    val1..n is treated as singleton. If val1..n is a dict, then values are sampled between a minimum and maximum value.

    Parameters
    ----------
    d : dict
        Dict to use.

    Returns
    -------
    numpy.ndarray of dict
    """
    d = deepcopy(d)
    # adjust dict
    for key, val in d.items():
        if isinstance(val, dict):
            # sample between a minimum and maximum value
            d[key] = np.linspace(val["min"], val["max"], val["n"])
        elif not isinstance(val, Iterable):
            # if item is not iterable, wrap in list
            d[key] = [val]

    dp = list(itertools.product(*d.values()))
    return np.array([dict(zip(d.keys(), k)) for k in dp]).reshape(tuple([len(val) for key, val in d.items()]))