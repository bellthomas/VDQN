import glob
import os 
from pathlib import Path
import re
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline
import matplotlib
import matplotlib.pyplot as plt
import datetime as dt

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{libertine}\usepackage[libertine]{newtxmath}\usepackage{sfmath}\usepackage[T1]{fontenc}'

dir_path = os.path.dirname(os.path.realpath(__file__))
experiments = {}
data_folder = "combined-data"
graphs_folder = "combined-graphs"

def extract_data(filename, variational=False):
    with open(filename, "r") as f:
        data = f.read()
        matches = []
        # Episode 8 (i: 4434, 2.56 seconds) --- r: -433.0 (avg: -492.55555555555554)
        patterns = [
            "((?<=Episode )\d+)",
            "((?<=i: )\d+(?=,))",
            "((?<=, )\d+.\d+(?= seconds))",
            "((?<=r: )\-?\d+.\d+(?= ))",
            "((?<=\(avg: )\-?\d+.\d+(?=\)))",
        ]
        if variational:
            patterns.append("((?<=\(vi: )\-?\d+.\d+|nan(?=,))")
            patterns.append("((?<=, bellman: )\-?\d+.\d+|nan(?=\)))")

        for pattern in patterns:
            matches.append([(float(x) if x != "nan" else -1) for x in re.findall(pattern, data)])

        return list(zip(*matches))

def generate_spline(xs, ys, errs):
    if (len(xs) == 0 or len(ys) == 0 or len(errs) == 0):
        return xs, ys, errs

    ys = np.array(ys)
    err = np.array(errs)

    x_vals = np.array(xs)
    xnew = np.linspace(x_vals.min(), x_vals.max(), len(xs) * 3) 
    # print(xnew)
    spl = make_interp_spline(x_vals, ys, k=2)  # type: BSpline
    spl_err = make_interp_spline(x_vals, err, k=2)  # type: BSpline
    ys_smooth = spl(xnew)
    err_smooth = spl_err(xnew)

    return xnew, ys_smooth, err_smooth

def uneven_tuple_zip(*lists, index=1):
    results = []
    for _l in lists:
        _index = 0
        for _e in _l:
            if(len(_e) > index):
                if(_index >= len(results)):
                    results.append([_e[index]])
                else:
                    results[_index].append(_e[index])
            _index +=1

    return results

def apply_averaging(l, averaging=10):
    result = []
    for index in range(len(l)):
        result.append(np.mean( l[max(0, index+1-averaging):index+1] ))
    return result

###########



base = "{}/{}".format(dir_path, data_folder)
experiments = {}
scenarios = [os.path.basename(x) for x in glob.glob("{}/*".format(base))]
for scenario in scenarios:
    algorithms = [os.path.basename(x) for x in glob.glob("{}/{}/*".format(base, scenario))]
    for algorithm in algorithms:
        runs = [os.path.basename(x) for x in glob.glob("{}/{}/{}/*".format(base, scenario, algorithm))]
        for run in runs:
            path = ("{}/{}/{}/{}".format(base, scenario, algorithm, run))
            if scenario not in experiments:
                experiments[scenario] = {}
            
            if algorithm not in experiments[scenario]:
                experiments[scenario][algorithm] = []

            exp_data = extract_data(path, variational=(algorithm in ["VDQN","DVDQN"]))
            experiments[scenario][algorithm].append(exp_data)

print("Data present:")
for exp in experiments:
    keys = experiments[exp].keys()
    _k = []
    for key in keys:
        _k.append("{} ({})".format(key, len(experiments[exp][key])))
    print("   {} - {}".format(exp, ", ".join(_k)))


print("\nDrawing graphs...")
graphs_dir = "{}/{}".format(dir_path, graphs_folder)
os.makedirs(graphs_dir, exist_ok=True)

# exit()

# Loops
averaging = 10
colours = {
    "DQN": "red",
    "DDQN": "orange",
    "VDQN": "blue",
    "DVDQN": "purple"
}
indices = [
    ("Reward", 3, False, "Reward"),
    ("VI-Loss", 5, True, "VI\ Loss"),
    ("Bellman-Loss", 6, True, "Bellman\ Loss"),
]
for exp in experiments:
    print(exp)
    for to_plot in indices:
        experiment = experiments[exp]
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 5), sharex=False, sharey=False)
        axes.margins(x=0)
        # axes.ylim(top=140000)
        axes.set_ylabel("${}$".format(to_plot[3]), fontsize=16)
        axes.set_xlabel("$Episodes$", fontsize=16)

        drawn = 0
        algorithms = []
        for algorithm in list(experiment.keys()):
            if to_plot[2] and algorithm not in ["VDQN","DVDQN"]:
                continue

            _d = uneven_tuple_zip(*experiments[exp][algorithm], index=to_plot[1])
            if(len(_d) > 2):
                algorithms.append(algorithm)
                _d_mu = apply_averaging([np.mean(x) for x in _d], averaging=averaging)
                _d_sigma = apply_averaging([np.std(x) for x in _d], averaging=averaging)
                _xs = range(1, len(_d)+1)
                xs, ys, errs = generate_spline(_xs, _d_mu, _d_sigma)

                drawn += 1
                axes.plot(xs, ys, 'k-', color=colours.get(algorithm, 'black'), alpha=0.8)
                axes.fill_between(xs, ys-errs, ys+errs, color=colours.get(algorithm, 'black'), alpha=0.25)
        
        axes.legend(algorithms)
        plt.tight_layout()
        if(drawn > 0):
            outdir = "{}/{}".format(graphs_dir, exp)
            os.makedirs(outdir, exist_ok=True)
            plt.savefig("{}/{}-wide.png".format(outdir, to_plot[0]), dpi=600)
            print("   {}/{}-wide.png ({} plots)".format(outdir, to_plot[0], drawn))

        plt.close(fig)
