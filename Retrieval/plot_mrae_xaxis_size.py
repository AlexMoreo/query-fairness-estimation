import itertools
import os.path
from Retrieval.experiments import methods
from Retrieval.commons import CLASS_NAMES, Ks, DATA_SIZES
import matplotlib.pyplot as plt

from Retrieval.plot_mrae_xaxis_k import load_all_results

data_home = 'data'
class_mode = 'multiclass'

method_names = [name for name, *other in methods(None)]

all_results = {}

class_name_label = {
    'continent': 'Geographic Location',
    'gender': 'Gender',
    'years_category': 'Age of Topic'
}

# loads all MRAE results, and returns a dictionary containing the values, which is indexed by:
# class_name -> data_size -> method_name -> k -> stat -> float
results = load_all_results()

# generates the class-independent, size-independent plots for y-axis=MRAE in which:
# - the x-axis displays the Ks

# X_DATA_SIZES = [int(x.replace('K', '000').replace('M', '000000').replace('FULL', '3250000')) for x in DATA_SIZES]
X_DATA_SIZES = [x.replace('FULL', '3.25M') for x in DATA_SIZES]

for class_name in CLASS_NAMES:
    for k in [100]: #Ks:

        log = class_name=='gender'

        fig, ax = plt.subplots()

        max_means = []
        markers = itertools.cycle(['o', 's', '^', 'D', 'v', '*', '+'])
        for method_name in method_names:
            # class_name -> data_size -> method_name -> k -> stat -> float
            means = [
                results[class_name][data_size][method_name][k]['mean'] for data_size in DATA_SIZES
            ]
            stds = [
                results[class_name][data_size][method_name][k]['std'] for data_size in DATA_SIZES
            ]
            # max_mean = np.max([
            #         results[class_name][data_size][method_name][k]['max'] for data_size in DATA_SIZE
            # ])

            max_means.append(max(means))

            style = 'o-' if method_name != 'CC' else '--'
            method_name = method_name.replace('NaiveQuery', 'Naive@$k$')
            method_name = method_name.replace('KDEy-ML', 'KDEy')
            marker=next(markers)
            line = ax.plot(X_DATA_SIZES, means, style, label=method_name, color=None, linewidth=3, markersize=10, marker=marker)
            color = line[-1].get_color()
            if log:
                ax.set_yscale('log')
            # ax.fill_between(Ks, means - stds, means + stds, alpha=0.3, color=color)

        ax.grid(True, which='both', axis='y', color='gray', linestyle='--', linewidth=0.3)
        ax.set_xlabel('training pool size')
        ax.set_ylabel('RAE' + (' (log scale)' if log else ''))
        ax.set_title(f'{class_name_label[class_name]} at exposure {k=}')
        ax.set_ylim([0, max(max_means)*1.05])

        if class_name == 'years_category':
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        os.makedirs(f'plots/var_size/{class_name}', exist_ok=True)
        plotpath = f'plots/var_size/{class_name}/{k}_mrae.pdf'
        print(f'saving plot in {plotpath}')
        plt.savefig(plotpath, bbox_inches='tight')











