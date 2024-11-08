import itertools
import os.path
import pickle
import numpy as np
from Retrieval.experiments import methods
from Retrieval.commons import CLASS_NAMES, Ks, DATA_SIZES
from os.path import join
import matplotlib.pyplot as plt



data_home = 'data'
class_mode = 'multiclass'

method_names = [name for name, *other in methods(None, 'continent')]

all_results = {}

class_name_label = {
    'continent': 'Geographic Location',
    'gender': 'Gender',
    'years_category': 'Age of Topic'
}


# loads all MRAE results, and returns a dictionary containing the values, which is indexed by:
# class_name -> data_size -> method_name -> k -> stat -> float
# where stat is "mean", "std", "max"
def load_all_results():

    for class_name in CLASS_NAMES:

        all_results[class_name] = {}

        for data_size in DATA_SIZES:

            all_results[class_name][data_size] = {}

            results_home = join('results', class_name, class_mode, data_size)

            all_results[class_name][data_size] = {}

            for method_name in method_names:
                results_path = join(results_home, method_name + '.pkl')
                try:
                    results = pickle.load(open(results_path, 'rb'))
                except Exception as e:
                    print(f'missing result {results}', e)

                all_results[class_name][data_size][method_name] = {}
                for k in Ks:
                    all_results[class_name][data_size][method_name][k] = {}
                    values = results['mrae']
                    all_results[class_name][data_size][method_name][k]['mean'] = np.mean(values[k])
                    all_results[class_name][data_size][method_name][k]['std'] = np.std(values[k])
                    all_results[class_name][data_size][method_name][k]['max'] = np.max(values[k])

    return all_results


results = load_all_results()

# generates the class-independent, size-independent plots for y-axis=MRAE in which:
# - the x-axis displays the Ks

for class_name in CLASS_NAMES:
    for data_size in DATA_SIZES[:1]:

        log = class_name=='gender'

        fig, ax = plt.subplots()

        max_means = []
        markers = itertools.cycle(['o', 's', '^', 'D', 'v', '*', '+'])
        for method_name in method_names:
            # class_name -> data_size -> method_name -> k -> stat -> float
            means = [
                results[class_name][data_size][method_name][k]['mean'] for k in Ks
            ]
            stds = [
                results[class_name][data_size][method_name][k]['std'] for k in Ks
            ]
            # max_mean = np.max([
            #         results[class_name][data_size][method_name][k]['max'] for k in Ks
            # ])
            max_means.append(max(means))

            means = np.asarray(means)
            stds = np.asarray(stds)

            method_name = method_name.replace('NaiveQuery', 'Naive@$k$')
            method_name = method_name.replace('KDEy-ML', 'KDEy')
            marker = next(markers)
            line = ax.plot(Ks, means, 'o-', label=method_name, color=None, linewidth=3, markersize=10, marker=marker)
            color = line[-1].get_color()
            if log:
                ax.set_yscale('log')
            # ax.fill_between(Ks, means - stds, means + stds, alpha=0.3, color=color)

        ax.grid(True, which='both', axis='y', color='gray', linestyle='--', linewidth=0.3)
        ax.set_xlabel('k')
        ax.set_ylabel('RAE' + (' (log scale)' if log else ''))
        data_size_label = '$\mathcal{L}_{10\mathrm{K}}$'
        ax.set_title(f'{class_name_label[class_name]} from {data_size_label}')
        ax.set_ylim([0, max(max_means)*1.05])

        if class_name == 'years_category':
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        os.makedirs(f'plots/var_k/{class_name}', exist_ok=True)
        plotpath = f'plots/var_k/{class_name}/{data_size}_mrae.pdf'
        print(f'saving plot in {plotpath}')
        plt.savefig(plotpath, bbox_inches='tight')











