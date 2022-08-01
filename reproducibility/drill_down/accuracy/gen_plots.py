import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c']
hatches = [4*x for x in ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']]
markers = ['o', '^', '*', '<']


def gen_plots():
    approaches = ['gpu-no-mat-no-fuse', 'gpu-mat-fuse']
    cycles = list(range(1, 11))
    approaches_display_names = ['Current Practice', 'Nautilus']

    iters = ['1', '2', '3']
    labelling_times = [0.0, 4]
    labelling_time_plot_names = ['A', 'B']
    batch_num_records = 500

    for j, labelling_time in enumerate(labelling_times):
        values = []
        times = []
        begin_times = []

        for i, approach in enumerate(approaches):
            for cycle in cycles:
                for iteration in iters:
                    log_fname = "../../end_to_end/ftr2/{}-{}.log".format(iteration, approach)
                    lines = open(log_fname).readlines()
                    begin_time = datetime.strptime(lines[0].split(': ')[0].split('=>')[-1].strip(), '%Y-%m-%d %H:%M:%S.%f')

                    lines = [l for l in open(log_fname).readlines() if 'Cycle: {}, Best model val_accuracy'.format(cycle) in l]
                    assert len(lines) == 1
                    
                    values.append(eval(lines[0].split('val_accuracy:')[-1].strip())[-1])
                    times.append(((datetime.strptime(lines[0].split(': ')[0].split('=>')[-1].strip(), '%Y-%m-%d %H:%M:%S.%f') - begin_time).total_seconds() + (cycle+1)*batch_num_records*labelling_time)/60.0)


        values = np.array(values)
        values = values.reshape([len(approaches), len(cycles), len(iters)])
        values_mean = np.mean(values, axis=2)
        values_std = np.std(values, axis=2)


        times = np.array(times)
        times = times.reshape([len(approaches), len(cycles), len(iters)])
        times = np.mean(times, axis=2)
        
        fig, ax = plt.subplots(figsize=(4,3.5))
        ax.plot(times[0], values_mean[0], label=approaches_display_names[0], marker=markers[0], color=colors[1])
        ax.plot(times[1], values_mean[1], label=approaches_display_names[1], marker=markers[1], color=colors[1])

        ax.legend()
        ax.set_xlabel('Elapsed Time (min)')
        ax.set_ylabel('Best Model Val. Accuracy')
        plt.grid()
        fig.tight_layout()

        plt.savefig('./Figure_7{}.pdf'.format(labelling_time_plot_names[j]), dpi=300, bbox_inches = 'tight', pad_inches = 0)


if __name__ == "__main__":
    gen_plots()
