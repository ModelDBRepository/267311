# Implements various models
#
# Copyright (c) 2022 Adrian Negrean
# negreanadrian@gmail.com
#
# Software released under MIT license, see license.txt for conditions

# for naming model functions, use this convention:
# _set_model_<type> or _set_model_<type><num>
# when a model is called, first the specific numbered model is called if implemented, next the generic model and finally if none are present,
# then only section properties or mechanisms are inserted and applied.

import os, sys
from collections.abc import Iterable
import numpy as np

import neuron as nrn
from neuron import h as nrnh
min_nrn_ver = (7, 8, 2)
if not tuple(map(int, nrn.version.split('.'))) >= min_nrn_ver:
    raise EnvironmentError(
        "NEURON version too low, required at least {}".format('.'.join(str(n) for n in min_nrn_ver)))

from nrnengine import nrn_util as nutil, util

def call_model(mname, mset, env_set):
    """
    Dispatches model calls, prioritizing a particular model implementation as <model name><model num>
    over generic model implementation <model_name>
    """
    # apply segment or section parameters
    if "segspec" in mset:
        if isinstance(mset["segspec"], dict):
            nutil.set_seg(h = nrnh, par = mset["segspec"], sec_list = env_set["seclists"], subtree_sec_list = env_set["subtree_seclists"])
        elif isinstance(mset["segspec"], Iterable):
            for ss in mset["segspec"]:
                nutil.set_seg(h = nrnh, par = ss, sec_list = env_set["seclists"], subtree_sec_list = env_set["subtree_seclists"])
        else:
            raise TypeError()

    if mname in models:
        getattr(sys.modules[__name__],"_set_model_"+mname)(h = nrnh, mset = mset, env_set = env_set)
        return
    for mn in models:
        if mname.startswith(mn):
            getattr(sys.modules[__name__],"_set_model_"+mn)(h = nrnh, mset = mset, env_set = env_set)
            return

# checked 8/19/21
def _set_model_ENV(h, mset, env_set):
    """
    Sets simulation physiological environment.

    Parameters
    ----------
    h : NEURON hoc.HocObject
        NEURON HOC interpreter object.
    mset : dict
        Model settings.
    env_set : dict
        Simulation engine settings. Must have
        'morph_dir' : str key specified with absolute path to folder storing neuronal morphologies.

    Returns
    -------
    None
    """
    # set temperature
    h.celsius = mset["temp"]

    # set initial global ionic concentrations in [mM]
    # note: some ions may not be defined in hoc if there are no mechanism using them
    for iname, ic in mset["ions"].items():
        iname0 = "{}0_{}_ion".format(iname, iname[:-1])
        if iname0 in vars(h):
            setattr(h, iname0, ic)

# checked 08/01/21
def _set_model_NM(h, mset, env_set):
    """
    Sets neuronal morphologies.

    Parameters
    ----------
    h : NEURON hoc.HocObject
        NEURON HOC interpreter object.
    mset : dict
        Model settings.
    env_set : dict
        Simulation engine settings. Must have
        'morph_dir' : str key specified with absolute path to folder storing neuronal morphologies.

    Returns
    -------
    None
    """
    nutil.load_hoc_file(h, os.path.join(env_set["morph_dir"], mset["morph"]))
    print("Morphology contains {} sections.".format(nutil.get_nsecseg(h = nrnh)[0]))
    # load section lists defined in hoc and defined in the workflow JSON to the environment
    if "seclists" in mset:
        for sl_name, sl_val in mset["seclists"].items():
            # try to load section list from HOC
            if sl_val is None:
                # check if section list exists in hoc
                if not hasattr(nrnh, sl_name):
                    raise Exception("'{}' section list has not been defined in HOC.".format(sl_name))
                sl = list(getattr(nrnh, sl_name))
                assert all([isinstance(s, nrn.nrn.Section) for s in sl])
                env_set["seclists"][sl_name] = sl
            else:
                # assemble section list
                env_set["seclists"][sl_name] = []
                for s in sl_val:
                    env_set["seclists"][sl_name].append(nutil.get_secseg(h = nrnh, secseg = s))
    if "seglists" in mset:
        for sl_name, sl_val in mset["seglists"].items():
            env_set["seglists"][sl_name] = []
            for s in sl_val:
                env_set["seglists"][sl_name].append(nutil.get_secseg(h = nrnh, secseg = s))
    # build section lists that belong to a subtree defined by a parent section
    if "subtree_seclists" in mset:
        for sl_name, sl_val in mset["subtree_seclists"].items():
            if isinstance(sl_val["parent_sec"], str):
                parent_secs = [sl_val["parent_sec"]]
            else:
                parent_secs = sl_val["parent_sec"]

            env_set["subtree_seclists"][sl_name] = []
            for psec in parent_secs:
                if sl_val["include_parent"]:
                    secs = [nutil.get_secseg(h = h, secseg = psec)]
                else:
                    secs = nutil.get_secseg(h = h, secseg = psec).children()
                for s in secs:
                    env_set["subtree_seclists"][sl_name].append(s.subtree())

            # flatten and add to section lists as well
            env_set["seclists"][sl_name] = util.flatten_list(env_set["subtree_seclists"][sl_name])

def _set_model_SCM(h, mset, env_set):
    """
    Creates a single section compartment model.

    Parameters
    ----------
    h : NEURON hoc.HocObject
        NEURON HOC interpreter object.
    mset : dict
        Model settings.
    env_set : dict
        Simulation engine settings.

    Returns
    -------
    None
    """
    util.set_default_keys(
        {
            "nseg": 1,
            "L": 20,
            "diam": 20
        }, mset)
    h.execute("create {}".format(mset["sec_name"]))
    sec = getattr(h, mset["sec_name"])
    sec.nseg = mset["nseg"]
    sec.L = mset["L"]
    for seg in sec:
        seg.diam = mset["diam"]

def _set_model_CCM(h, mset, env_set):
    """
    Creates a chained compartment model.

    Parameters
    ----------
    h : NEURON hoc.HocObject
        NEURON HOC interpreter object.
    mset : dict
        Model settings.
    env_set : dict
        Simulation engine settings.

    Returns
    -------
    None
    """
    # create compartments
    for sec_name, sec_info in mset.items():
        h.execute("create {}".format(sec_name))
        sec = getattr(h, sec_name)
        util.set_default_keys(
            {
                "nseg": 1,
                "L": 20,
                "diam": 20
            }, sec_info)
        sec.nseg = sec_info["nseg"]
        sec.L = sec_info["L"]
        for seg in sec:
            seg.diam = sec_info["diam"]
    # connect compartments
    sec_names = list(mset.keys())
    for sn_idx in range(len(sec_names)-1):
        sec_parent = getattr(h, sec_names[sn_idx])
        sec_child = getattr(h, sec_names[sn_idx+1])
        sec_child.connect(sec_parent(1))

def _set_model_IC(h, mset, env_set):
    """
    Inserts current clamp point processes.
    """
    for ic_name, ic_spec in mset["iclamps"].items():
        env_set["pproc"][ic_name] = nutil.IClamp(seg = ic_spec["seg"], waveform = ic_spec["waveform"])

def _set_model_NSEG(h, mset, env_set):
    """
    Use to print out model info.
    """
    print("Number of segments in each section:")
    for sec in sorted(h.allsec(), key = lambda x: x.name()):
        print("{} = {}".format(sec.name(), sec.nseg))

def _set_model_TEST(h, mset, env_set):
    """
    For testing
    """
    print([(seg.CaR.n, seg.x) for seg in h.trunk])
    raise Exception()

    
# list of available models (call this after all definitions)
models = [g[11:] for g in globals() if g.startswith("_set_model_")]