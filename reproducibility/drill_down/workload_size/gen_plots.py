import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c']
hatches = [4*x for x in ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']]

def gen_plots():
    exps = ['ftr3-1', 'ftr3-2', 'ftr3-3', 'ftr3-4', 'ftr3-5']

    approaches = ['mat-fuse', 'no-mat-fuse', 'mat-no-fuse']
    approaches_display_names = ['Nautilus', 'Nautilus w/o MAT OPT', 'Nautilus w/o FUSE OPT']

    iters = ['1', '2', '3']

    values = []   
    for approach in approaches:
        for exp in exps: 
            for iteration in iters:
                log_fname = "./logs/{}-{}-{}-{}.log".format(iteration, 'gpu', exp, approach)
                lines = [l for l in open(log_fname).readlines() if 'Workload completed. Elapsed time:' in l]
                assert len(lines) == 1
                workload_time = datetime.strptime(lines[0].split('Elapsed time:')[-1].strip(), '%H:%M:%S.%f') - datetime(1900, 1, 1)
                workload_time = workload_time.total_seconds()/60.0
                
                values.append(workload_time)

    values = np.array(values)
    values = values.reshape([len(approaches),len(exps), len(iters)])
    mean_values = np.mean(values, axis=-1)
    std_values = np.std(values, axis=-1)


    fig, ax = plt.subplots(figsize=(6.3,2.8))
    
    ind = np.arange(len(exps))   # the x locations for the groups
    width = 0.15                 # the width of the bars

    for i, approach in enumerate(approaches):
        ax.bar(ind+width*i, mean_values[i], width, bottom=0, yerr=4.303*std_values[i], label=approaches_display_names[i], hatch=hatches[i], color=colors[i])

    ax.set_xticks(ind + width)
    ax.set_xticklabels(['1', '2', '3', '4', '5'])
    ax.legend(ncol=1, frameon=True)
    
    ax.set_ylabel('Runtime (minutes)')
    ax.set_xlabel('# Models')
    plt.grid()
    fig.tight_layout()

    plt.savefig('./Figure_9.pdf', dpi=300, bbox_inches = 'tight', pad_inches = 0)

            
if __name__ == "__main__":
    gen_plots()
