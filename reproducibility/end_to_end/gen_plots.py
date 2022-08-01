import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

colors = ['#a6cee3','#1f78b4','#33a02c','#b2df8a']
hatches = [4*x for x in ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']]


def gen_e2e_plots1():
    exps = ['ftr1', 'ftr2', 'ftr3', 'atr', 'ftu']

    approaches = ['no-mat-no-fuse', 'mat-all-no-fuse', 'mat-fuse']
    approaches_display_names = ['Current Practice', 'MAT-ALL', 'Nautilus', 'FLOPs Optimal']
    theoretical_speedups = [1/4.11, 1/4.62, 1/4.02, 1/3.19, 1/2.87]

    iters = ['1', '2', '3']

    values = []   
    for approach in approaches:
        for exp in exps: 
            for iteration in iters:
                log_fname = "./{}/{}-{}-{}.log".format(exp, iteration, 'gpu', approach)
                lines = [l for l in open(log_fname).readlines() if 'Workload completed. Elapsed time:' in l]
                assert len(lines) == 1
                workload_time = datetime.strptime(lines[0].split('Elapsed time:')[-1].strip(), '%H:%M:%S.%f') - datetime(1900, 1, 1)
                workload_time = workload_time.total_seconds()/60.0
                values.append(workload_time)

    values = np.array(values)
    values = values.reshape([len(approaches),len(exps), len(iters)])
    mean_values = np.mean(values, axis=-1)
    mean_values = np.vstack([mean_values, mean_values[:1,:] * np.array(theoretical_speedups)])

    std_values = np.std(values, axis=-1)
    std_values = np.vstack([std_values, np.array([[0, 0, 0, 0, 0]])])


    fig, ax = plt.subplots(figsize=(6,2.5))
    
    ind = np.arange(len(exps))   # the x locations for the groups
    width = 0.15                 # the width of the bars

    for i, approach in enumerate(approaches+['theoretical_optimal']):
        ax.bar(ind+width*i, mean_values[i], width, bottom=0, yerr=4.303*std_values[i], label=approaches_display_names[i], hatch=hatches[i], color=colors[i])

    ax.set_xticks(ind + 3/2*width)
    ax.set_xticklabels(['FTR-1', 'FTR-2', 'FTR-3', 'ATR', 'FTU'])
    ax.legend(ncol=2, frameon=True)
    ax.set_ylim(0, 800)
    
    ax.set_xlabel('Workload')
    plt.grid()
    fig.tight_layout()

    plt.savefig('./Figure_6A.pdf', dpi=300, bbox_inches = 'tight', pad_inches = 0)


def gen_e2e_plots2():
    device = 'gpu'
    exp = 'ftr2'
    approaches = ['no-mat-no-fuse', 'mat-fuse']
    approaches_display_names = ['Current Practice', 'Nautilus']

    iters = ['1', '2', '3']
    values = []
    for approach in approaches:
        for iteration in iters:
            log_fname = "./{}/{}-{}-{}.log".format(exp, iteration, device, approach)
            lines = open(log_fname).readlines()
            
            model_initalization_time = [l for l in lines if 'Completed system initialization. Elapsed time:' in l]
            assert len(model_initalization_time) == 1
            model_initalization_time = datetime.strptime(model_initalization_time[0].split('Elapsed time:')[-1].strip(), '%H:%M:%S.%f') - datetime(1900, 1, 1)
            model_initalization_time = model_initalization_time.total_seconds()/60.0
            values.append(model_initalization_time)

            model_selection_iter_times = [l for l in lines if 'Completed active learning cycle. Elapsed time:' in l]
            for i, l in enumerate(model_selection_iter_times):
                if i%2 != 0:
                    continue

                model_selection_iter_time = datetime.strptime(l.split('Elapsed time:')[-1].strip(), '%H:%M:%S.%f') - datetime(1900, 1, 1)
                model_selection_iter_time = model_selection_iter_time.total_seconds()/60.0
                values.append(model_selection_iter_time)

    values = np.array(values)
    values = values.reshape([len(approaches),len(iters), -1])
    mean_values = np.mean(values, axis=1)
    std_values = np.std(values, axis=1)

    fig, ax = plt.subplots(figsize=(3,2.5))
    ind = np.arange(mean_values.shape[1])   # the x locations for the groups
    width = 0.32                             # the width of the bars

    for i, approach in enumerate(approaches):
        ax.bar(ind+width*i, mean_values[i], width, bottom=0, yerr=4.303*std_values[i], label=approaches_display_names[i], hatch=hatches[i+i*1], color=colors[i+i*1])

    ax.set_xticks(ind + width * 1/2)
    ax.set_xticklabels(['Init'] + [i*2+1 for i in list(range(mean_values.shape[1]-1))])
    ax.legend(ncol=1, frameon=True)
    ax.set_ylabel('Runtime (mins)')
    ax.set_xlabel('Model Selection Cycle')
    plt.grid()
    ax.set_ylim(0, 80)

    plt.savefig('./Figure_6B.pdf', dpi=300, bbox_inches = 'tight', pad_inches = 0)



def gen_e2e_plots3():
    labelling_times = [0.5, 1, 2, 4, 8]

    approaches = ['no-mat-no-fuse', 'mat-fuse']
    approaches_display_names = ['Current Practice', 'Nautilus']

    iters = ['1', '2', '3']
    num_records = 5000

    values = []   
    for approach in approaches:
        for labelling_time in labelling_times: 
            for iteration in iters:
                log_fname = "./{}/{}-{}-{}.log".format('ftr2', iteration, 'gpu', approach)
                lines = [l for l in open(log_fname).readlines() if 'Workload completed. Elapsed time:' in l]
                assert len(lines) == 1
                workload_time = datetime.strptime(lines[0].split('Elapsed time:')[-1].strip(), '%H:%M:%S.%f') - datetime(1900, 1, 1)
                workload_time = (workload_time.total_seconds() + labelling_time * num_records)/60.0
                
                values.append(workload_time)

    values = np.array(values)
    values = values.reshape([len(approaches),len(labelling_times), len(iters)])
    mean_values = np.mean(values, axis=-1)

    std_values = np.std(values, axis=-1)

    fig, ax = plt.subplots(figsize=(3,2.5))
    ind = np.arange(mean_values.shape[1])   # the x locations for the groups
    width = 0.32                             # the width of the bars

    for i, approach in enumerate(approaches):
        ax.bar(ind+width*i, mean_values[i], width, bottom=0, yerr=4.303*std_values[i], label=approaches_display_names[i], hatch=hatches[i+i*1], color=colors[i+i*1])

    ax.set_xticks(ind + 1/2*width)
    ax.set_xticklabels(labelling_times)
    ax.legend(ncol=1, frameon=True, loc='upper left')
    ax.set_ylim(0, 1200)
    
    ax.set_ylabel('Total Workload Time (mins)')
    ax.set_xlabel('Labelling Time Per Record (secs)')
    plt.grid()
    fig.tight_layout()

    plt.savefig('./Figure_6C.pdf', dpi=300, bbox_inches = 'tight', pad_inches = 0)



if __name__ == "__main__":
    gen_e2e_plots1()
    gen_e2e_plots2()
    gen_e2e_plots3()
