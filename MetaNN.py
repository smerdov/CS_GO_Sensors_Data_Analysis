import os
import argparse
# from subprocess import Popen
from multiprocessing import Pool
import itertools
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--super-suffix', type=str,
                    default='last_exp')

# super_suffix = 'attention_multiplier_2_0'  # 'march_6'
# time_step_list = [5, 10, 20, 30]  # [5, 10, 20, 30, 40]
# window_size_list = [60, 120, 180, 300]  # [60, 120, 180, 300]
# time_step_list = [5, 10, 20, 30]  # [5, 10, 20, 30, 40]
time_step_list = [20]  # [5, 10, 20, 30, 40]
window_size_list = [180]  # [60, 120, 180, 300]
batch_size_list = [1024]  # 32
hidden_size_list = [8]
attention_list = [1]  # [4, 0, 2, 1] [0, 1, 2, 3, 4, 5, 6, 7, 8]
n_attention_layers_list = [1]  # Looks like 2 is better than 1
# n_dense_layers_list = [1, 2, 3, 4]  # 3  # Looks like 2 is a bit better than 1
n_dense_layers_list = [2]  # 3  # Looks like 2 is a bit better than 1
# n_attention_layers_with_hidden_state_list = [0, 1]
n_attention_layers_with_hidden_state_list = [1]
# cell_type_list = ['gru']  # 'no_cell', 'rnn',
cell_type_list = ['gru']  # 'no_cell', 'rnn',
# cell_type_list = ['no_cell']  # 'no_cell', 'rnn',
opt_list = ['adam']
normalization_list = [0]

n_repeat = 40
n_inits = 40
plot_attention = 5


n_processes = 8
scorer = 'auc'  # 'auc'
target_types_list = ['more_than_avg_fair']
# target_types_list = ['more_than_avg']
# target_types_list = ['more_than_avg']
every_step_training_list = [1]
# target_types_list = ['more_than_median', 'more_than_avg', 'more_than_before']

# attention_multiplier = 3
warmup = 10  # 30
n_epoches = 500
# batches4epoch = 20  # 20
max_patience = 3  # 30
hidden_state_warmup = 0
plot_cell = False
plot_predict = False
plot_target = False
modify_attention_array = False
recreate_dataset = 0
target_prefix = 'kills_proportion'


player_ids = ['9', '0', '11', '7', '6', '1', '10', '19', '8', '21', '4', '3', '12', '2', '5', '14', '22'] + ['13', '15', '16', '17']


def list2cmd_format(x):
    if type(x) == list:  # As expected
        x = [str(y) for y in x]
        return ' '.join(x)
    else:
        return x  # Probably there is only one value

args = parser.parse_args()
print(f'super_suffix is {args.super_suffix}')

def run_command(time_step, window_size, batch_size, hidden_size, attention, normalization, n_attention_layers,
                n_dense_layers, n_attention_layers_with_hidden_state, cell_type, opt_list, target_types, every_step_training):
    # super_suffix = args.super_suffix + f'_{process_id}'
    super_suffix = args.super_suffix
    command = f'python NN.py '\
              f'--super_suffix {super_suffix} '\
              f'--loss bce --scorer {scorer} '\
              f'--max_patience {max_patience} '\
              f'--plot_attention {plot_attention} '\
              f'--recreate_dataset {recreate_dataset} '\
              f'--n_repeat {n_repeat} '\
              f'--player_ids {list2cmd_format(player_ids )} '\
              f'--time_step_list {list2cmd_format(time_step)} '\
              f'--window_size_list {list2cmd_format(window_size)} '\
              f'--batch_size_list {list2cmd_format(batch_size)} '\
              f'--hidden_size_list {list2cmd_format(hidden_size)} '\
              f'--attention_list {list2cmd_format(attention)} '\
              f'--normalization_list {list2cmd_format(normalization)} '\
              f'--n_attention_layers_list {list2cmd_format(n_attention_layers)} '\
              f'--n_dense_layers_list {list2cmd_format(n_dense_layers)} '\
              f'--n_attention_layers_with_hidden_state_list {list2cmd_format(n_attention_layers_with_hidden_state)} '\
              f'--cell_type_list {list2cmd_format(cell_type)} '\
              f'--opt_list {list2cmd_format(opt_list)} '\
              f'--target_types {list2cmd_format(target_types)} '\
              f'--arch=splitted_nn '\
              f'--every_step_training={every_step_training} '\
              f'--n_inits={n_inits} '\
              f'--warmup {warmup} '# \
              # f'--append_dumb_predict'
              # f'--target_verbose '

    os.system(command)

if __name__ == '__main__':
    pool_args = itertools.product(time_step_list, window_size_list, batch_size_list, hidden_size_list, attention_list, normalization_list,
                      n_attention_layers_list, n_dense_layers_list, n_attention_layers_with_hidden_state_list, cell_type_list, opt_list, target_types_list, every_step_training_list)

    pool_args = list(pool_args)
    pool_args = np.random.permutation(pool_args)  # I'm increasing the probability having the error soon after start
    # rather than in the end
    print(f'Conducting {len(pool_args)} experiments, each with {n_repeat} repetitions')
    print(pool_args)

    pool = Pool(n_processes)
    pool.starmap(run_command, pool_args)


#
# p = Popen(os.path.join(os.getcwd(), 'NN.py'), shell=True)
# p.wait()
#
#
# print(command)
#
# # os.system(command)
#
#

