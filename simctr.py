# Copyright (c) 2022 Adrian Negrean
# negreanadrian@gmail.com
#
# Software released under MIT license, see license.txt for conditions
import sys
min_python_ver = (3, 8, 0)
if not sys.version_info >= min_python_ver:
    raise EnvironmentError(
        "Python version too low, required at least {}".format('.'.join(str(n) for n in min_python_ver)))

import os, argparse, pprint, commentjson, re, shutil, pickle as pkl
from datetime import datetime 
from copy import deepcopy
from multiprocessing import cpu_count, Pool
from functools import partial
import numpy as np

import neuron as nrn
from neuron import h as nrnh
min_nrn_ver = (7, 8, 2)
if not tuple(map(int, nrn.version.split('.'))) >= min_nrn_ver:
    raise EnvironmentError(
        "NEURON version too low, required at least {}".format('.'.join(str(n) for n in min_nrn_ver)))

from nrnengine import nrn_models as nmod, nrn_util as nutil, morphology as nmorph, util, plots as plts

from matplotlib import pyplot as plt
import matplotlib as mpl
#mpl.use('Agg')
# change font type to 42 to be able to edit text in Adobe Illustrator
mpl.rcParams['pdf.fonttype'] = 42
from matplotlib.backends.backend_pdf import PdfPages
# ------------------------------------ GLOBALS ------------------------------------
# absolute path to folder where this script is
this_script_dir = os.path.dirname(os.path.abspath(__file__))

def _expand_secseg_list(secsegs, env_set):
    """
    Expand lists containing section and segment lists.

    Parameters
    ----------
    secsegs : list
        List containing section lists with names starting with "~" and segment lists with names starting with "#".

    env_set : dict
        Simulation environment settings.

    Returns
    -------
    list of str
        Expanded sections and segment names.
    """
    out = []
    for x in secsegs:
        # section lists
        if x[0] == "~":
            out.extend([nutil.get_hoc_name(h = nrnh, obj = obj) for obj in env_set["seclists"][x[1:]]])
        # segment lists
        elif x[0] == "#":
            out.extend([nutil.get_hoc_name(h = nrnh, obj = obj) for obj in env_set["seglists"][x[1:]]])
        # sections and segments
        else:
            out.append(x)

    return out

def _model_worker(ps, tset, env_set):
    """
    NEURON model worker process.

    Parameters
    ----------
    env_set : dict
        Simulation wnvironment settings.
    tset : dict
        Run task settings.
    ps : dict
        Model space parameter modifiers.

    Returns
    -------
    run-data : dict
        Simulation output.
    """
    # switch to use variable time-step integration
    cvode = nrnh.CVode()
    cvode.active(tset["use_cvode"])

    # check model names
    # model names must start with [a-zA-Z_]+, and may end with numbers
    model_names = tset["mspec"].split('-')
    mn_chk = [re.match("[a-zA-Z_]+[0-9]*", mn) for mn in model_names]
    if any([m is None for m in mn_chk]):
        raise Exception("Model names must start with [a-zA-Z_]+, and may end with numbers.")

    # collect run data together with model modifiers
    run_data = {}
    # measurement data
    run_data["rec"] = {}
    # add model specification
    run_data["mspec"] = tset["mspec"]
    # make a copy of all models and then modify parameters
    mod_models = deepcopy(env_set["workflow"]["models"])
    util.set_dpath(ps, mod_models)
    # collect modified model space that is relevant to the current model specification
    run_data["mspace"] = {m_name:mod_models[m_name] for m_name in model_names if m_name in mod_models}
    # keep track of what parameters have been modified on this run
    run_data["pmod"] = ps

    # load HOC files
    if "load_hoc" in tset:
        for hoc_rel_path in tset["load_hoc"]:
            nutil.load_hoc_file(h = nrnh, fpath = os.path.join(this_script_dir, hoc_rel_path))

    # filter priority model names: morphology, capacitance, axial resistivity
    # WARNING: if adding new morphology models, they must be added here as priority models
    priority_models = ["NM", "SCM", "CCM", "C", "RA"]
    priority_model_idxs = []
    for pm in priority_models:
        mn_match = "{}[0-9]+".format(pm)
        for idx, mn in enumerate(model_names):
            if re.match(mn_match, mn) is not None:
                priority_model_idxs.append(idx)
    priority_model_names = [model_names[mn_idx] for mn_idx in priority_model_idxs]
    # remove priority models from original list
    model_names = util.del_list_idxs(model_names, priority_model_idxs)

    # filter environment models
    # note: environment ionic concentrations must be applied after all models since some
    # ions are added only if mechanisms need them
    env_model_idxs = []
    for idx, mn in enumerate(model_names):
        if re.match("ENV[0-9]+", mn) is not None:
            env_model_idxs.append(idx)
    env_model_names = [model_names[mn_idx] for mn_idx in env_model_idxs]
    # remove environment models from original list
    model_names = util.del_list_idxs(model_names, env_model_idxs)

    # apply priority models
    for mname in priority_model_names:
        nmod.call_model(mname = mname, mset = run_data["mspace"][mname], env_set = env_set)
    # resegment (this needs to be done after axial resistivity, capacitance have been set)
    nutil.resegment(h = nrnh, freq = tset["d_lambda_freq"], d_lambda = tset["d_lambda"], maxLseg = tset["maxLseg"], use_d_lambda = tset["use_d_lambda"])
    # load remaining models
    for mname in model_names:
        nmod.call_model(mname = mname, mset = run_data["mspace"][mname], env_set = env_set)
    # apply environment models
    for mname in env_model_names:
        nmod.call_model(mname = mname, mset = run_data["mspace"][mname], env_set = env_set)

    # ------------------------------------ prepare recording vectors ------------------------------------------------
    if "rec" in tset:
        secseg_rec_spec = {} # recording specification for sections and segments
        pp_rec_spec  = {} # recording specification for point processes
        for rec_parname, rec_targets in tset["rec"].items():
            partype, parname = rec_parname.split(":")
            if partype == "seg":
                # expand here section and segment lists
                secseg_rec_spec[parname] = _expand_secseg_list(secsegs = rec_targets, env_set = env_set)
            elif partype == "pp":
                # convert point process names to objects
                pp_rec_spec[parname] = []
                for pp_name in rec_targets:
                    if pp_name in env_set["pproc"]:
                        pp_rec_spec[parname].append(env_set["pproc"][pp_name].pproc)
                    else:
                        raise Exception("Point process '{}' was not defined in the simulation environment.".format(pp_name))
            else:
                raise ValueError("Recording parameter type '{}' not implemented.".format(partype))
        secseg_rec_spec = nutil.record_hoc_par(h = nrnh, recinfo = secseg_rec_spec)
        pp_rec_spec = nutil.record_hoc_par(h = nrnh, recinfo = pp_rec_spec)
        conversion_ppnames = {pp.pproc.hname(): user_name for user_name, pp in env_set["pproc"].items()}
        # convert hoc named point processes to user names
        adjusted_pp_rec_spec = {}
        for rec_parname, hoc_ppnames in pp_rec_spec.items():
            adjusted_pp_rec_spec[rec_parname] = {conversion_ppnames[k]:v for k,v in hoc_ppnames.items()}

        # combine recording vectors
        recv = {"seg:"+k:v for k,v in secseg_rec_spec.items()}
        recv.update({"pp:"+k:v for k,v in adjusted_pp_rec_spec.items()})
    else:
        recv = {}
    
    # ------------------------------------------ run model ------------------------------------------------------------
    nrnh.dt = tset["dt"]
    nrnh.finitialize(tset["vinit"])
    nrnh.fcurrent()

    if recv:
        nrn.run(tset["simdur"])

    # ------------------------------------ convert recording vectors to numpy arrays ----------------------------------
    for rec_parname, hoc_objs in recv.items():
        run_data["rec"][rec_parname] = {}
        for hoc_obj_name, recvs in hoc_objs.items():
            run_data["rec"][rec_parname][hoc_obj_name] = np.array(recvs)

    # ------------------------------------ measure impedance ----------------------------------
    # store segment impedance data in run_data/zin/seg:zin:/<secseg name> as 3D numpy nd.array with 0-index = test freq, 1-index = segment number, 2-index |Zin| in MOhm
    # and <Zin in [deg]
    if "zin" in tset:
        if "secseg" in tset["zin"]:
            secsegs = _expand_secseg_list(secsegs = tset["zin"]["secseg"] , env_set = env_set)
        else:
            secsegs = [sec.name() for sec in nrnh.allsec()]
        run_data["zin"] = {}
        for sname in secsegs:
            run_data["zin"][sname] = np.concatenate([nutil.meas_Zin_freq_domain(h = nrnh, freq = zin_freq, obj = nutil.get_secseg(h = nrnh, secseg = sname))[None,:] \
                for zin_freq in tset["zin"]["freq"]])

    # --- store dendrogram distance info ----
    run_data["dtree"] = nmorph.get_tree(h = nrnh, cb_func = lambda sec: np.array([seg.x for seg in sec])*sec.L)

    return run_data

# add more tasks as
# def _task_<task name>
def _task_run_model(tset, env_set, output_dir):
    """
    Builds and runs one or more NEURON models from multiple submodels.

    Parameters
    ----------
    tset : dict
        Model building task settings. Dict with keys:
            'mspec' : str
                Model name, string with submodel names separated by '-', e.g. 'NM1-RA1' for a morphology and axial
                resistivity specification.    
    env_set : dict
        Simulation environment settings.
    output_dir : str
        Task output folder.

    Returns
    -------
    None
    """
    # set default task parameters
    util.set_default_keys(
        {
            "use_cvode": False,
            "d_lambda_freq": 1000,
            "d_lambda": 0.1,
            "maxLseg": 2,
            "use_d_lambda": True,
            "psweep": {}
        }, tset)
    # store simulation time step so that e.g. point processes can generate waveforms with correct sampling.
    env_set["dt"] = tset["dt"]
    # sweep over models by modifying the model space
    if tset["psweep"]:
        env_set["psweep"] = util.dict_product(tset["psweep"])
    else:
        env_set["psweep"] = np.array([{}])

    with Pool(processes = env_set["nproc"]) as pool:
       env_set["mruns"] = pool.map(partial(_model_worker, tset = tset, env_set = env_set), env_set["psweep"].ravel())
    env_set["mruns"] = np.atleast_1d(np.array(env_set["mruns"]).reshape(env_set["psweep"].shape))
    
def _task_plot_recpar(tset, env_set, output_dir):
    """
    Plot recording parameters organized by their source within a grid. For parameter sweeps,
    the last parameter sweep axis can be used to plot waveforms using a color palette gradient.
    
    Parameters
    ----------
    tset : dict
        Task settings.
    env_set : dict
        Simulation environment settings.
    output_dir : str
        Task output folder.

    Returns
    -------
    None
    """
    # iterate over plot settings
    figures = []
    for pset in tset:
        # iterate over data to plot
        for rd in pset["recdata"]:
            if not rd["recpar"] in env_set["mruns"].flat[0]["rec"]:
                raise Exception("Plotting parameter '{}' was not recorded.".format(rd["recpar"]))
            
            figures.append(plts.rec_grid_plot(pset = pset, mruns = env_set["mruns"], recpar = rd["recpar"], dt = env_set["dt"],
                secseg_names_filter = rd["secseg"]))

    # save plots
    for idx, fig in enumerate(figures):
        # set default figure name
        util.set_default_keys(
            {
                "plot_base_name": "recpar_plot{}.pdf".format(idx+1)
            }, tset[idx])
        # name of recorded parameters plot
        output_pdf = os.path.join(output_dir, tset[idx]["plot_base_name"] if tset[idx]['plot_base_name'].endswith('.pdf') else tset[idx]['plot_base_name']+'.pdf') 
        with PdfPages(output_pdf) as pdf:    
            pdf.savefig(fig, orientation = "landscape", transparent = True)

    plt.show()

def _task_plot_zin(tset, env_set, output_dir):
    """
    Plot input impledance dendrogram.

    Parameters
    ----------
    tset : dict
        Task settings.
    env_set : dict
        Simulation environment settings.
    output_dir : str
        Task output folder.

    Returns
    -------
    None
    """
    # set defaults
    util.set_default_keys(
        {
            "alpha": 1
        }, tset)
    recdim = env_set["mruns"].shape
    assert len(recdim)<=2
    # use a grid plot
    fig, ax = plt.subplots(recdim[0], recdim[1], squeeze = False, sharex = True)
    for row_idx in range(recdim[0]):
        for col_idx in range(recdim[1]):
            # dendrogram data
            dtree = env_set["mruns"][row_idx, col_idx]["dtree"] # tuple
            zin = env_set["mruns"][row_idx, col_idx]["zin"]
            # plot dendrogram for selected morphology root section
            plts.plot_dendrogram(dtree = {tset["root_sec_name"]: dtree[tset["root_sec_name"]]}, secdata = zin,
                ax = ax[row_idx, col_idx], alpha = tset["alpha"])
            # add x axis label to bottom plots
            if row_idx == recdim[0]-1:
                ax[row_idx, col_idx].set_xlabel("Distance to root ($\mu$m)") # path distance
        # add y axis label
        ax[row_idx, 0].set_ylabel("|Zin| (M$\Omega$)") # path distance

    plt.show()
    
def _task_print_zin(tset, env_set, output_dir):
    """
    Prints input impedance of section as 2D array with rows ordered by the segment within sections, first columns as |Zin| in MOhm and
    second column as <Zin phase in deg.

    Parameters
    ----------
    tset : dict
        Task settings.
    env_set : dict
        Simulation environment settings.
    output_dir : str
        Task output folder.

    Returns
    -------
    None
    """
    recdim = env_set["mruns"].shape
    for row_idx in range(recdim[0]):
        for col_idx in range(recdim[1]):
            print("========= ({}, {}) =========")
            zin = env_set["mruns"][row_idx, col_idx]["zin"]
            pp = pprint.PrettyPrinter(indent = 4)
            pp.pprint(zin)

def main(argv):
    """
    NEURON model simulation control.
    """
    # environment settings
    env_set = {}
    # store both section lists defined in hoc and in python
    env_set["seclists"] = {}
    # store segment lists defined in python
    env_set["seglists"] = {}
    # store section list subtrees as list of list of sections with first section in the list being the root of the subtree
    env_set["subtree_seclists"] = {}
    # store point processes, with dict keys being names assigned to the point processes such as current clamps
    env_set["pproc"] = {}
    # stores NEURON model simulation data as "model runs" over a given parameter space sweep
    env_set["mruns"] = np.array([], dtype = "object")
    # simulation environment time step in [ms] set by run_model
    env_set["dt"] = None
    # keep track of parameter sweeps
    env_set["psweep"] = np.array([], dtype = "object")

    # list of available task types
    task_types = [g[6:] for g in globals() if g.startswith("_task_")]
    
    # parse command line arguments
    argParser = argparse.ArgumentParser()
    argParser.add_argument("settings", action = "store", type = str, help = "Simulation settings as JSON file path and name.")
    argParser.add_argument("task", action = "store", type = str, help = "Simulation pipeline to use.")
    argParser.add_argument("-p", "--nproc", action = "store", type = int, default = cpu_count(), help = "Number of processes to use. Defaults to maximum number of CPUs.")
    args = argParser.parse_args(argv)

    # get current execution timestamp
    run_timestamp = datetime.now().strftime("%y%m%d_%H%M%S")

    # create output folder to save recorded parameters and other plots
    output_dir = os.path.join(this_script_dir, "task-output", args.task+"_"+run_timestamp)

    # load workflow
    with open(args.settings) as f:
        # read key value pairs as OrderedDict to make processing more predictable (since keys in a normal dict are not retrieved in an ordered way)
        env_set["workflow"] = commentjson.load(f) # use object_pairs_hook = OrderedDict if order is not preserved
        if args.task in env_set["workflow"]["tasks"]:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        else:
            raise ValueError("Invalid task name '{}'.".format(args.task))

    try:
        # copy settings file to output folder
        shutil.copy(args.settings, output_dir)
        
        # convert to absolute paths relative to this script
        env_set["morph_dir"] = os.path.join(this_script_dir, env_set["workflow"]["nrn_morph"])
        env_set["mech_dir"] = os.path.join(this_script_dir, env_set["workflow"]["nrn_mech"]) if "nrn_mech" in env_set["workflow"] else ""

        # load external membrane mechanisms
        if env_set["mech_dir"]:
            nutil.load_mech(h = nrnh, fpath = env_set["mech_dir"])    

        # set maximum number of processes
        env_set["nproc"] = args.nproc
        # call task steps
        for ts in env_set["workflow"]["tasks"][args.task]:
            # separate task name from task type
            task_type = ts.split(":")[0]
            if task_type in task_types:
                getattr(sys.modules[__name__],"_task_"+task_type)(tset = env_set["workflow"]["task-steps"][ts], env_set = env_set, output_dir = output_dir)
            else:
                raise Exception("Task type '{}' is not implemented.".format(task_name))

        # save recorded parameters
        with open(os.path.join(output_dir, "mruns.pickle"), 'wb') as f:
            pkl.dump(env_set["mruns"], f)

    except Exception as e:
        # clear all contents if exception occurs
        shutil.rmtree(output_dir)
        raise e

if __name__ == '__main__':
    main(sys.argv[1:])