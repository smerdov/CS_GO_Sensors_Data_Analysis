




all_attentions_list = []
for model_name, attentions4players in best_attentions_dict.items():
    for player_id, attentions4player in attentions4players.items():
        all_attentions_list = all_attentions_list + [attentions4player]#.reshape(15, 1)


# mean_att = np.mean(attention_sum_list_dict['30_300_16_32_2_1_3_prefinal'], axis=0)
mean_att = np.mean(all_attentions_list[::], axis=0)
index_order = np.argsort(mean_att)
mean_att = np.median(all_attentions_list[::], axis=0)
index_order = np.argsort(mean_att)


color_att = 'olivedrab'
color_att = 'olive'
color_att = 'darkslategrey'
color_att = 'darkcyan'

margin = 0.018
fontsize = 14
plt.close()

plt.figure(figsize=(8, 5))
y_ticks = list(range(len(index_order)))
plt.barh(y_ticks, mean_att[index_order], color=color_att)
plt.xlim((mean_att.min() - margin, mean_att.max() + margin * 0.7))
plt.yticks(y_ticks, np.array(features_pretty)[index_order], fontsize=fontsize)
plt.xlabel('Mean Attention', fontsize=fontsize+3)
plt.title('Feature Importance', fontsize=fontsize+6)
plt.tight_layout()
plt.savefig('pic/attention_importance_v0.png')


plt.interactive(True)


plt.barh(mean_att[index_order], np.array(features_pretty)[index_order])



# for time_step in [5, 10, 20, 30, 40, 60, 120]:
#     command = f'python TimeseriesFinalPreprocessing.py --TIMESTEP {time_step}'
#     os.system(command)
#     print('Done')
#



