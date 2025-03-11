#!/usr/bin/env python3
import os
import multiprocessing as mp
import numpy as np
import awkward as ak
import uproot

from tools.roottree import read_tree
from tools.selections import *


rich_selectors = {"LIP": {"Tof": selector_tof, "NaF": selector_naf_lipvar, "Agl": selector_agl_lipvar},
             "CIEMAT": {"Tof":selector_tof, "NaF": selector_naf_ciematvar, "Agl": selector_agl_ciematvar}}

def initialize_tree(file, treename, branch_type_dict):
    file.mktree(treename, branch_type_dict, title=treename)

def handle_file(arg):
    filename, treename, chunk_size, rank, nranks, kwargs = arg
    
    variables = kwargs["variables"]
    resultdir = kwargs["resultdir"]
    outputprefix = kwargs["outputprefix"]
    verbose = kwargs["verbose"]
    nbytes_min = kwargs["nbytes_min"]
    qsel = kwargs["qsel"]
    datatype = kwargs["datatype"]
    isotope = kwargs["isotope"]
    nuclei = kwargs["nuclei"]
    isIss = kwargs["isIss"]
    betatype = kwargs["betatype"]
    detector = kwargs["detector"]
    read_all_branches = variables is None

    if read_all_branches:
        branches = None
    else:
        branches = set(variables)
        all_derived_branches = set()

    event_cache = None
    trees_initialized = False

    with uproot.recreate(os.path.join(resultdir, f"{outputprefix}_{rank}.root")) as root_file:
        for events in read_tree(filename, treename, branches=branches, rank=rank, nranks=nranks, chunk_size=chunk_size, verbose=verbose):
            if not trees_initialized:
                if read_all_branches:
                    variables = events.fields
                if int(ak.__version__.split(".")[0]) == 1:
                    branch_types = dict(events.type.type)
                else:
                    branch_types = dict(zip(events.type.content.fields, events.type.content.contents))
                                        
                irregular_branches = [branch for branch in variables if type(branch_types[branch]).__name__ == "ListType"]
                counter_branches = [f"n{branch_name}" for branch_name in irregular_branches]
                var_type_dict = {var: branch_types[var] for var in variables}
                initialize_tree(root_file, treename, var_type_dict)
                trees_initialized = True

            for branch in irregular_branches:
                events = ak.with_field(events, ak.num(events[branch]), f"n{branch}")

            #events = ak.packed(events)
            #events = rich_selectors[betatype][detector](events, nuclei, isotope, datatype)
            events = remove_badrun_indst(events) 
            events = SelectCleanEvent(events)
            events = events[events.is_ub_l1 == 1]
            events = rich_selectors[betatype][detector](events, nuclei, isotope, datatype, cutoff=True)
            
            if len(events) == 0:
                continue

            reduced_array = ak.Array({branch_name: events[branch_name] for branch_name in variables + counter_branches})
            if event_cache is None:
                event_cache = reduced_array
            else:
                event_cache = ak.concatenate((event_cache, reduced_array))

            if event_cache is not None and event_cache.nbytes >= nbytes_min and len(event_cache) > 0:
                root_file[treename].extend(event_cache)
                event_cache = None

        if event_cache is not None:
            root_file[treename].extend(event_cache)


def make_args(filename, treename, chunk_size, nranks, parallel, **kwargs):
    parallel_index, parallel_total = parallel
    assert parallel_index >= 0 and parallel_index < parallel_total
    for rank in range(parallel_index * nranks, (parallel_index + 1) * nranks):
        yield (filename, treename, chunk_size, rank, nranks * parallel_total, kwargs)


def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--filename", nargs="+", default="trees/", required=True, help="Path to tree file(s).")
    parser.add_argument("--treename", default="amstreea", help="Tree name in file.")
    parser.add_argument("--outputprefix", default="TreeRichAgl", help="Prefix for the reduced trees.")
    parser.add_argument("--resultdir", default="trees", help="Directory to store the reduced trees in.")
    parser.add_argument("--nprocesses", type=int, default=1, help="Number of parallel processes.")
    parser.add_argument("--parallel", type=int, nargs=2, default=(0, 1), help="Index of this job and total number of parallel jobs.")
    parser.add_argument("--chunk-size", type=int, default=200000, help="Number of evetns per chunk to read.")
    parser.add_argument("--nbytes-min", type=int, default=int(10e6), help="Minimum number of bytes to write a basket.")
    parser.add_argument("--variables", nargs="+", help="Variables to store in the reduced trees.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Output verbose progress information.")
    parser.add_argument("--qsel", required=True, type=float, help="given the charge of the particle that would be selected.")
    parser.add_argument("--datatype", default="ISS", help="datatype: MC or ISS.")
    parser.add_argument("--isotope", default="Be7", help="isotope id for calculate the geomanetic cutoff")
    parser.add_argument("--nuclei", default="Be", help="nuclei type")
    parser.add_argument("--isIss", default=False, type=bool, help="data type")
    parser.add_argument("--betatype", default="CIEMAT", help="LIP or CIMEAT")
    parser.add_argument("--detector", default="Agl", help="Tof or NaF or Agl")

    args = parser.parse_args()    
    parallel = args.parallel
    parallel_index, parallel_total = parallel

    if args.variables is None:
        print("Writing all branches.")

    os.makedirs(args.resultdir, exist_ok=True)

    with mp.get_context("fork" if args.verbose else "spawn").Pool(args.nprocesses) as pool:
        pool_args = make_args(args.filename, args.treename, args.chunk_size, args.nprocesses, parallel, variables=args.variables, resultdir=args.resultdir, outputprefix=args.outputprefix, nbytes_min=args.nbytes_min, verbose=args.verbose, qsel=args.qsel, datatype=args.datatype, isotope=args.isotope, nuclei=args.nuclei, isIss=args.isIss, betatype=args.betatype, detector=args.detector)

        for rank in pool.imap_unordered(handle_file, pool_args):
            pass


if __name__ == "__main__":
    main()
