import pandas as pd

pd.options.display.max_columns = 15

df_players = pd.read_csv('data/participants.csv', sep=',')

# df_players.columns

emotion_columns_before_raw = ['Interested (заинтересованный)', 'Distressed (обеспокоенный)',
       'Excited (возбужденный, воодушевленный)', 'Upset (расстроенный)',
       'Strong (сильный, уверенный)', 'Guilty (виновный)',
       'Scared (испуганный)', 'Hostile (враждебный)',
       'Enthusiastic (с энтузиазмом)', 'Proud (гордящийся)',
       'Irritable (раздражительный)', 'Alert (тревожный)',
       'Ashamed (стыдящийся)', 'Inspired (вдохновленный)',
       'Nervous (неврничающий)', 'Determined (определившийся)',
       'Attentive (внимательный)', 'Jittery (невный, пугливый)',
       'Active (активный)', 'Afraid (испуганный)']

emotion_columns_after_raw = [column + '.1' for column in emotion_columns_before_raw]
emotion_columns_after = [column + '_after' for column in emotion_columns_before_raw]
emotion_columns_before = [column + '_before' for column in emotion_columns_before_raw]

rename_dict = dict(zip(emotion_columns_before_raw + emotion_columns_after_raw, emotion_columns_before + emotion_columns_after))

rename_dict.update({
    ' What experience do u have in shooter games (Counter-Strike, Doom, Battlefield, etc.)?': 'Skill',
    'How much hours did you spent playing Counter-Strike (all versions)?': 'Hours exp',
    'Id': 'player_id',
})

df_players.rename(columns=rename_dict,inplace=True)


columns2drop = ['Timestamp',
                'When was the last time you played a shooter game?',
                'When was the last time you played Counter-Strike?',
                'What is your current rank in CS:GO?',
                'What was your maximal rank in CS:GO?',
                'Choose the picture that best suits how you feel',
                'Choose a pattern that better matches the intensity of your feeling',
                'Choose the picture that best suits how you feel.1',
                'Choose a pattern that better matches the intensity of your feeling.1',
                'Was was your mouse sensitivity in game?',
]

df_players.drop(columns2drop, axis=1, inplace=True)

# df_players['Hours'].sort_values()
hours_border_list = [100, 1000]

for hours_border in hours_border_list:
    # feature_name = f'More than {hours_border} hours experience'
    feature_name = f'>{hours_border} h exp'
    df_players[feature_name] = (df_players['Hours exp'] >= hours_border) * 1

# player_features = ['player_id', 'Skill', 'Hours', 'Gender', 'Age'] + emotion_columns_after
# df_players = df_players[player_features]

skill_is_none = df_players['Skill'] == 'None'
df_players.loc[skill_is_none, 'Skill'] = 'Small'
df_players['player_id'] = df_players['player_id'].astype(str)

df_players['Gender'] = pd.factorize(df_players['Gender'])[0]  # 0 --- male, 1 --- female
df_players['Skill'] = pd.factorize(df_players['Skill'])[0]

df_players.to_csv('data/players.csv', index=False)







