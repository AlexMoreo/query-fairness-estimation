import os.path
from Retrieval.experiments import methods
from Retrieval.commons import CLASS_NAMES, Ks, DATA_SIZES
import matplotlib.pyplot as plt

from Retrieval.plot_mrae_xaxis_k import load_all_results

data_home = 'data'
class_mode = 'multiclass'

method_names = [name for name, *other in methods(None)]

all_results = {}


# loads all MRAE results, and returns a dictionary containing the values, which is indexed by:
# class_name -> data_size -> method_name -> k -> stat -> float
results = load_all_results()

# generates the class-independent, size-independent plots for y-axis=MRAE in which:
# - the x-axis displays the Ks

for class_name in CLASS_NAMES:
    for k in Ks:

        log = True

        fig, ax = plt.subplots()

        max_means = []
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
            line = ax.plot(DATA_SIZES, means, style, label=method_name, color=None)
            color = line[-1].get_color()
            if log:
                ax.set_yscale('log')
            # ax.fill_between(Ks, means - stds, means + stds, alpha=0.3, color=color)

        ax.set_xlabel('training pool size')
        ax.set_ylabel('RAE' + ('(log scale)' if log else ''))
        ax.set_title(f'{class_name} from {k=}')
        ax.set_ylim([0, max(max_means)*1.05])

        ax.legend()

        os.makedirs(f'plots/var_size/{class_name}', exist_ok=True)
        plotpath = f'plots/var_size/{class_name}/{k}_mrae.pdf'
        print(f'saving plot in {plotpath}')
        plt.savefig(plotpath, bbox_inches='tight')











