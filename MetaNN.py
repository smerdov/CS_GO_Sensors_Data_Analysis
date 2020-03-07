import os

# criterion = nn.BCELoss()  # nn.MSELoss()
# scorer = roc_auc_score  # accuracy_score, mean_squared_error
# classification = (scorer == roc_auc_score) or (scorer == accuracy_score)

time_step_list = [5]  # [5, 10, 20, 30, 40]
window_size_list = [300]  # [60, 120, 180, 300]
batch_size_list = [512]
hidden_size_list = [64]
attention_list = [2, 4]  # [4, 0, 2, 1] [0, 1, 2, 3, 4, 5, 6, 7, 8]
super_suffix = 'attention_multiplier_2_0'  # 'march_6'
normalization_list = [0]  # , 1]  # [0, 1]
n_repeat = 5
attention_multiplier = 3

n_epoches = 200
batches4epoch = 20  # 20
max_patience = 10
hidden_state_warmup = 10
plot_cell = False
plot_predict = False
plot_attention = 30
plot_target = False
modify_attention_array = False

recreate_dataset = 0
target_prefix = 'kills_proportion'


player_ids = ['9', '0', '11', '7', '6', '1', '10', '19', '8', '21', '4', '3', '12', '2', '5', '14', '22'] #  + ['13', '15', '16', '17']


def list2cmd_format(x):
    x = [str(y) for y in x]
    return ' '.join(x)


command = f'python NN.py '\
          f'--super_suffix {super_suffix} '\
          f'--loss bce --scorer auc '\
          f'--max_patience {max_patience} '\
          f'--plot_attention {plot_attention} '\
          f'--recreate_dataset {recreate_dataset} '\
          f'--time_step_list {list2cmd_format(time_step_list)} '\
          f'--window_size_list {list2cmd_format(window_size_list)} '\
          f'--batch_size_list {list2cmd_format(batch_size_list)} '\
          f'--hidden_size_list {list2cmd_format(hidden_size_list)} '\
          f'--attention_list {list2cmd_format(attention_list)} '\
          f'--normalization_list {list2cmd_format(normalization_list)} '\
          f'--player_ids {list2cmd_format(player_ids)} '\
          f'--n_repeat {n_repeat} '\
          f'--attention_multiplier {attention_multiplier}'


os.system(command)



