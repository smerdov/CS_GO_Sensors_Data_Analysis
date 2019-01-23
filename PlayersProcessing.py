import pandas as pd

df_players = pd.read_csv('data/participants.csv', sep=',')

# df_players.columns

emotion_columns_before = ['Interested (заинтересованный)', 'Distressed (обеспокоенный)',
       'Excited (возбужденный, воодушевленный)', 'Upset (расстроенный)',
       'Strong (сильный, уверенный)', 'Guilty (виновный)',
       'Scared (испуганный)', 'Hostile (враждебный)',
       'Enthusiastic (с энтузиазмом)', 'Proud (гордящийся)',
       'Irritable (раздражительный)', 'Alert (тревожный)',
       'Ashamed (стыдящийся)', 'Inspired (вдохновленный)',
       'Nervous (неврничающий)', 'Determined (определившийся)',
       'Attentive (внимательный)', 'Jittery (невный, пугливый)',
       'Active (активный)', 'Afraid (испуганный)']

emotion_columns_after = [column + '.1' for column in emotion_columns_before]

df_players.rename(columns={
    ' What experience do u have in shooter games (Counter-Strike, Doom, Battlefield, etc.)?': 'Skill',
    'How much hours did you spent playing Counter-Strike (all versions)?': 'Hours',
    'Id': 'player_id',
    },
    inplace=True,
)

player_features = ['player_id', 'Skill', 'Hours', 'Gender', 'Age'] + emotion_columns_after

df_players = df_players[player_features]
skill_is_none = df_players['Skill'] == 'None'
df_players.loc[skill_is_none, 'Skill'] = 'Small'
df_players['player_id'] = df_players['player_id'].astype(str)

df_players.to_csv('data/players.csv', index=False)



