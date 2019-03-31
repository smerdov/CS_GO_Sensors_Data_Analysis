import sys
import os
import itertools
import argparse

# session_duration_list = [int(x * 60) for x in [1, 2, 5, 10]]
# session_duration_list = [int(x * 60) for x in [2, 3, 4]]
# session_duration_list = [int(x * 60) for x in [3, 5]]
session_duration_list = [int(x * 60) for x in [3]]
max_sessions_per_player = [10]
# interval_list = [0.06, 0.25, 1, 4]
# interval_list = [3, 5]
# interval_list = [1, 2]
interval_list = [1]
shift_list = [1]
# reaction_multiplier_list = [2, 3.5, 5]
reaction_multiplier_list = [5]

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='', type=str)

if __name__ == '__main__':  # add params later
    args = parser.parse_args()
    name = args.name

    for session_duration, max_sessions_per_player, interval, shift, reaction_multiplier in \
            itertools.product(session_duration_list, max_sessions_per_player, interval_list, shift_list, reaction_multiplier_list):

        params_str_list = [str(x) for x in [session_duration // 60, max_sessions_per_player, interval, shift, reaction_multiplier]]
        suffix = f'_{name}' + '_'.join(params_str_list)
        # os.system('python GeneralDataProcessing.py')
        os.system(f'python Main.py --session_duration {session_duration} '
                  f'--max_sessions_per_player {max_sessions_per_player} '
                  f'--interval {interval} --shift {shift} --reaction_multiplier {reaction_multiplier} '
                  f'--suffix {suffix}')




