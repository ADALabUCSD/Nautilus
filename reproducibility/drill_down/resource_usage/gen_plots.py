import numpy as np
from datetime import datetime
import re
import matplotlib.pyplot as plt

colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c']
hatches = [4*x for x in ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']]
markers = ['o', '^', '*', '<']


def disk_io_drill_down():
    approaches = ['gpu-no-mat-no-fuse', 'gpu-mat-fuse']

    read_values = [[],[]]
    write_values = [[],[]]
    begin_times = []
    end_times = []

    for approach in approaches:
        log_fname = "../../end_to_end/ftr2/{}-{}.log".format(1, approach)
        lines = open(log_fname).readlines()
        begin_times.append(datetime.strptime(lines[0].split(': ')[0].split('=>')[-1].strip(), '%Y-%m-%d %H:%M:%S.%f'))
        end_times.append(datetime.strptime(lines[-2].split(': ')[0].split('=>')[-1].strip(), '%Y-%m-%d %H:%M:%S.%f'))

    for i, approach in enumerate(approaches):
        log_fname = "./1_disk_utilization.log"

        lines = open(log_fname).readlines()
        if len(lines) % 2 != 0:
            lines = lines[:-1]

        for index in range(0, len(lines), 2):
            if begin_times[i] <= datetime.strptime(lines[index].strip(), '%Y-%m-%d %H:%M:%S') <= end_times[i]:
                l = lines[index+1]
                l = re.sub('\s+',' ',l).strip()
                write_values[i].append(float(l.split(' ')[-1]))
                read_values[i].append(float(l.split(' ')[-2]))

    for i in [0, 1]:
        read_utilization = np.array(read_values[i])
        write_utilization = np.array(write_values[i])
        
        # Averaging for a minutes. Raw values are at 2s intervals
        if read_utilization.shape[0]%30 != 0:
            read_utilization = read_utilization[:-(read_utilization.shape[0]%30)]
            write_utilization = write_utilization[:-(write_utilization.shape[0]%30)]
        
        read_utilization = read_utilization.reshape(-1, 30)[:,0]/1024**2 # GB
        read_utilization = read_utilization - read_utilization[0]
        write_utilization = write_utilization.reshape(-1, 30)[:,0]/1024**2 # GB
        write_utilization = write_utilization - write_utilization[0]
        
        print(read_utilization[-1], write_utilization[-1])

        full_utilization = np.stack([write_utilization, read_utilization])
        x = list(range(full_utilization.shape[1]))
        
        if i==0:
            fig, ax = plt.subplots(figsize=(8,2.5))
        else:
            fig, ax = plt.subplots(figsize=(8/5*1.2,2.5))
        
        ax.stackplot(x, full_utilization, colors=[colors[1], colors[0]], labels=['Write', 'Read'])
        ax.set_ylim(0, 150)
        ax.tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            left=False,      # ticks along the bottom edge are off
            right=False,         # ticks along the top edge are off
            labelleft=False) # labels along the bottom edge are off

        if i==0:
            ax.legend(loc='upper left')
            ax.set_xlim(0, 430)
        else:
            ax.set_xlim(0, 85)

        plt.grid()
        fig.tight_layout()

        plt.savefig('./Figure_11{}_1.pdf'.format(['A', 'B'][i]), dpi=300, bbox_inches = 'tight', pad_inches = 0)



def gpu_utilization_drill_down():
    approaches = ['gpu-no-mat-no-fuse', 'gpu-mat-fuse']

    values = [[],[]]
    begin_times = []
    end_times = []

    for approach in approaches:
        log_fname = "../../end_to_end/ftr2/{}-{}.log".format(1, approach)
        lines = open(log_fname).readlines()
        begin_times.append(datetime.strptime(lines[0].split(': ')[0].split('=>')[-1].strip(), '%Y-%m-%d %H:%M:%S.%f'))
        end_times.append(datetime.strptime(lines[-2].split(': ')[0].split('=>')[-1].strip(), '%Y-%m-%d %H:%M:%S.%f'))

    for i, approach in enumerate(approaches):
        log_fname = "./1_gpu_utilization.log"

        lines = open(log_fname).readlines()
        if len(lines) % 2 != 0:
            lines = lines[:-1]

        for index in range(0, len(lines), 2):
            if begin_times[i] <= datetime.strptime(lines[index].strip(), '%Y-%m-%d %H:%M:%S') <= end_times[i]:
                values[i].append(float(lines[index+1].split(',')[0].split(' ')[0]))


    for i in [0, 1]:
        utilization = np.array(values[i])
        # Averaging for a minutes. Raw values are at 2s intervals
        if utilization.shape[0]%30 != 0:
            utilization = utilization[:-(utilization.shape[0]%30)]
        
        utilization = utilization.reshape(-1, 30)
        utilization = np.mean(utilization, axis=-1)
        print(np.mean(utilization))

        if i==0:
            fig, ax = plt.subplots(figsize=(8,2.5))
        else:
            fig, ax = plt.subplots(figsize=(8/5*1.2,2.5))

        ax.plot(list(range(utilization.shape[0])), utilization, color=colors[1])
        ax.tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            left=False,      # ticks along the bottom edge are off
            right=False,         # ticks along the top edge are off
            labelleft=False) # labels along the bottom edge are off
            
        ax.set_ylim(0, 100)
        if i==0:
            ax.set_xlim(0, 430)
        else:
            ax.set_xlim(0, 85)

        plt.grid()
        fig.tight_layout()

        plt.savefig('./Figure_11{}_0.pdf'.format(['A', 'B'][i]), dpi=300, bbox_inches = 'tight', pad_inches = 0)


if __name__ == "__main__":
    gpu_utilization_drill_down()
    disk_io_drill_down()
