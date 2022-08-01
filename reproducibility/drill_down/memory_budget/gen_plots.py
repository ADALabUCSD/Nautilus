import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c']

hatches = [4*x for x in ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']]
hatches = [hatches[2]]

def gen_plots():
    device = 'gpu'
    approaches = ['no-mat-fuse']
    memory_budgets = [2, 4, 6, 8, 10]
    approaches_display_names = ['FUSE OPT']

    iters = ['1', '2', '3']
    values = []
    for approach in approaches:
        for mem in memory_budgets:
            for iteration in iters:
                log_fname = "logs/{}-{}-{}-{}.log".format(iteration, device, mem, approach)
                lines = [l for l in open(log_fname).readlines() if 'Workload completed. Elapsed time:' in l]
                assert len(lines) == 1
                workload_time = datetime.strptime(lines[0].split('Elapsed time:')[-1].strip(), '%H:%M:%S.%f') - datetime(1900, 1, 1)
                workload_time = workload_time.total_seconds()/60.0
                values.append(workload_time)
            

    values = np.array(values)
    values = values.reshape([len(approaches), len(memory_budgets), len(iters)])
    mean_values = np.mean(values, axis=-1)
    std_values = np.std(values, axis=-1)

    speedup_values = values[:,0:1,:]/values
    mean_speedup_values = np.mean(speedup_values, axis=-1)
    std_speedup_values = np.std(speedup_values, axis=-1)
    

    fig, ax = plt.subplots(figsize=(3,2.5))
    ind = np.arange(mean_values.shape[1])   # the x locations for the groups
    width = 0.5                             # the width of the bars

    ax.bar(ind, mean_values[0], width, bottom=0, yerr=4.303*std_values[0], label=approaches_display_names[0], color=colors[0])
    
    ax2 = ax.twinx()
    ax2.plot(ind, mean_speedup_values[0], colors[1], marker='o')
    ax2.set_ylabel('Speedup: line')
    ax2.set_ylim(0.0, 5.0)

    ax.set_xticks(ind)
    ax.set_xticklabels(memory_budgets)
    ax.set_ylabel('Runtime (minutes): bar')
    ax.set_xlabel('Runtime Memory Budget (GB)')
    plt.grid()
    fig.tight_layout()

    plt.savefig('./Figure_10B.pdf', dpi=300, bbox_inches = 'tight', pad_inches = 0)


if __name__ == "__main__":
    gen_plots()