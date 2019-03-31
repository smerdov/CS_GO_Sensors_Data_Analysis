import sys
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--session_duration', default=60 * 5, type=int)
parser.add_argument('--max_sessions_per_player', default=10, type=int)
parser.add_argument('--interval', default=3, type=float)
parser.add_argument('--shift', default=0, type=int)
parser.add_argument('--reaction_multiplier', default=5, type=float)
parser.add_argument('--suffix', default='', type=str)


if __name__ == '__main__':  # add params later
    args = parser.parse_args()
    session_duration = args.session_duration
    max_sessions_per_player = args.max_sessions_per_player
    interval = args.interval
    shift = args.shift
    suffix = args.suffix
    reaction_multiplier = args.reaction_multiplier

    # # os.system('python GeneralDataProcessing.py')
    # os.system(f'python SplittingToSessions.py --session_duration {session_duration} --max_sessions_per_player {max_sessions_per_player}')
    # os.system(f'python GameLogProcessing.py')
    # # os.system('python GeneralProcessing.py')
    os.system(f'python ChairProcessing.py --interval {interval} --shift {shift} --reaction_multiplier {reaction_multiplier}')
    # os.system('python ChairGamedataProcessing.py')
    os.system(f'python Merging.py')
    os.system(f'python Predicting.py --suffix {suffix}')







