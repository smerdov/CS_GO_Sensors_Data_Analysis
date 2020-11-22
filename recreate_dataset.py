import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--time_step_list', nargs='+', type=int)

def recreate_dataset_func(time_step):
    TIMESTEP_STRING = f'{time_step}s'
    print(TIMESTEP_STRING)
    print('recreating_part_0')
    command = f'python -O TimeseriesProcessing.py --TIMESTEP_STRING {TIMESTEP_STRING}'
    os.system(command)
    print('recreating_part_1')
    command = f'python -O TimeseriesMerging.py --TIMESTEP {time_step}'
    os.system(command)
    print('recreating_part_2')
    command = f'python -O TimeseriesAnalysis.py --TIMESTEP {time_step}'
    os.system(command)
    print('recreating_part_3')
    command = f'python -O TimeseriesFinalPreprocessing.py --TIMESTEP {time_step}'
    os.system(command)
    print(f'Dataset for {TIMESTEP_STRING} has been recreated')


if __name__ == '__main__':
    args = parser.parse_args()
    time_step_list = args.time_step_list

    for time_step in time_step_list:
        recreate_dataset_func(time_step)


