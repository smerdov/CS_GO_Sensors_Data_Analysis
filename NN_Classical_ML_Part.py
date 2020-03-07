
#########################
####### CLASSICAL ML ZONE
# train_test_splitter = StratifiedKFold(n_splits=5)
from hmmlearn import hmm

# time_step_list = [5, 10, 20, 30, 40]  # 10 is already tested
time_step_list = [30]  # 10 is already tested
# window_size_list = [60, 120, 180, 300, 600]
window_size_list = [300]
# window_size_list = [300]
n_repeat_list = list(range(5))
plot = False
results_list = []
rf_0 = RandomForestClassifier(n_estimators=100, max_depth=3)
# rf_1 = RandomForestClassifier(n_estimators=100, max_depth=5)
lr = LogisticRegression(solver='lbfgs')
svm = SVC(probability=True, gamma='auto')
# hmm_model_0 = hmm.GaussianHMM(n_components=10)  # , covariance_type="full")
# hmm_model_1 = hmm.GaussianHMM(n_components=3)  # , covariance_type="full")
# hmm_model_2 = hmm.GaussianHMM(n_components=5)  # , covariance_type="full")


alg_list = [lr, rf_0, svm] #, hmm_model_0, hmm_model_1, hmm_model_2]
alg_names_list = ['Logistic Regression', 'Random Forest', 'SVM'] # , 'Hidden Markov Model', 'Hidden Markov Model', 'Hidden Markov Model']

index_names = ['time_step', 'window_size', 'alg_name', 'n_repeat']
multi_index_all = pd.MultiIndex.from_product([time_step_list, window_size_list, alg_names_list, n_repeat_list], names=index_names)
df_results = pd.DataFrame(index=multi_index_all)
# df_results['score_train'] = -1
df_results['score_val'] = -1
suffix = 'window_size_0'

for time_step, window_size, n_repeat in itertools.product(time_step_list, window_size_list, n_repeat_list):
    # print(window_size)
    data_dict_resampled_merged_with_target_scaled = joblib.load(
                f'data/data_dict_resampled_merged_with_target_scaled_{int(time_step)}')

    target_prefix = 'kills_proportion'
    target_columns = [column for column in data_dict_resampled_merged_with_target_scaled['10'].columns if
                      column.startswith(target_prefix)]
    # window_size = 300
    target_column_past = f'{target_prefix}_{window_size}_4past'
    target_column_future = f'{target_prefix}_{window_size}_4future'

    train_tensors_dict = {}

    # for player_id, df4train in data_dict_resampled_merged_with_target_scaled.items():
    for player_id in player_ids:
        df4train = data_dict_resampled_merged_with_target_scaled[player_id]
        train_tensors4player = {}

        mask2keep = df4train[target_column_future].notnull() & df4train[target_column_past].notnull()

        if mask2keep.sum() == 0:
            print(f'Not enough data for player {player_id}')
            continue

        df4train = df4train.loc[mask2keep, :]
        df4train.fillna(0, inplace=True)

        target_future = df4train[target_column_future].values
        target_past = df4train[target_column_past].values
        #
        target = target_future - target_past
        # target_binary = (target_future < 0.1) * 1
        margin = 0  # 0
        target_binary = (target > margin) * 1
        # Possible targets:
        # better than average
        # 2 or 3 classes from: very bad, very good, average
        #

        # target = target_future

        # target_binary = (target > target.median()) * 1
        # target_binary.reset_index(drop=True, inplace=True)
        df4train.drop(columns=target_columns, inplace=True)
        df4train.reset_index(drop=True, inplace=True)
        features = list(df4train.columns)

        # if plot:
        #     plt.close()
        #     plt.plot(target_binary, label='target_binary')
        #     plt.plot(target, label='target')
        #     # plt.axhline(target.mean(), label='target_mean', color='green')
        #     plt.legend()
        #     plt.savefig(pic_folder + f'target_player_{player_id}')

        train_tensors4player['input'] = torch.Tensor(df4train.values)
        train_tensors4player['target'] = torch.Tensor(target_binary)  # FOR logloss metric
        train_tensors4player['target_raw'] = torch.Tensor(target)  # FOR logloss metric
        train_tensors4player['target_future'] = torch.Tensor(target_future)  # FOR logloss metric
        # train_tensors4player['target'] = torch.Tensor(target)  # FOR MSE metric
        # train_tensors4player['target'] = torch.Tensor(target_binary)
        # train_tensors4player['target_raw'] = torch.Tensor(target_binary)
        train_tensors_dict[player_id] = train_tensors4player


    train_test_splitter = KFold(n_splits=5, shuffle=True)
    players = list(train_tensors_dict.keys())
    # auc_scores_list = []


    for alg, alg_name in zip(alg_list, alg_names_list):
        # alg_name = alg.__class__.__name__
        auc_scores4alg = []
        dummy_scores = []


        if alg_name == 'Hidden Markov Model':
            ### For hidden states visualization for players
            for player_id in player_ids:
                plt.close()

                xx_train = train_tensors_dict[player_id]['input']
                yy_train = train_tensors_dict[player_id]['target_future']
                alg.fit(xx_train)
                predict_hard = alg.predict(xx_train)
                labels = np.unique(predict_hard)

                colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'orange', 'black', 'teal', 'brown']
                for label, color in zip(labels, colors):
                    indexes = np.nonzero(predict_hard == label)[0]
                    indexes = np.sort(indexes)
                    for index in indexes:
                        if index + 1 == len(predict_hard):
                            continue

                        x_data = [index, index + 1]
                        y_data = [yy_train[index], yy_train[index + 1]]

                        plt.plot(x_data, y_data, color=color)

                # plt.plot(predict_probas.argmax(axis=1), label='Predict')
                # plt.plot(target, label='Target')
                plt.title(alg_name)
                # plt.legend()
                plt.tight_layout()
                plt.savefig(f'pic/hmm_segmentation/{alg.n_components}_components_player_{player_id}_future.png')


        for train_players, test_players in train_test_splitter.split(players):
            # n_features = train_tensors_dict[players[0]]['input'].shape[1]
            # x_train = np.empty(shape=(0, n_features))
            # for train_player in train_players:
            x_train = np.concatenate([train_tensors_dict[players[player_id_index]]['input'] for player_id_index in train_players])
            y_train = np.concatenate([train_tensors_dict[players[player_id_index]]['target'] for player_id_index in train_players])


            # if alg_name == 'Hidden Markov Model':
            #     alg.fit(x_train)  # , [len(x_train)] * 15)
            #     # predict = alg.predict(x_val)
            #     # alg.predict_proba(x_val)
            #     # print(len(np.unique(predict)))
            # else:
            alg.fit(x_train, y_train)

            for test_player in [players[player_id_index] for player_id_index in test_players]:


                x_val = train_tensors_dict[test_player]['input']
                y_val = train_tensors_dict[test_player]['target']




                # y_val_raw = train_tensors_dict[test_player]['target_raw']
                if len(np.unique(y_val)) < 2:
                    continue
                else:
                    # predict_probas = alg.predict_proba(x_val)
                    # predict = predict_probas[:, 1]
                    predict = alg.predict(x_val)
                    # predict_hard = alg.predict(x_val)
                    dummy_shift = window_size // time_step
                    dummy_predict = [0] * dummy_shift + list(np.array(y_val).astype(int))[:-dummy_shift]
                    dummy_predict = np.array(dummy_predict)
                    dummy_predict = 1 - dummy_predict


                # if alg_name == 'Hidden Markov Model':
                #     plt.close()
                #
                #     labels = np.unique(predict_hard)
                #     colors = ['red', 'green', 'blue', 'yellow', 'magenta']
                #     for label, color in zip(labels, colors):
                #         indexes = np.nonzero(predict_hard == label)[0]
                #         indexes = np.sort(indexes)
                #         for index in indexes:
                #             if index + 1 == len(predict_hard):
                #                 continue
                #
                #             x_data = [index, index + 1]
                #             y_data = [y_val_raw[index], y_val_raw[index + 1]]
                #
                #             plt.plot(x_data, y_data, color=color)
                #
                #     # plt.plot(predict_probas.argmax(axis=1), label='Predict')
                #     # plt.plot(target, label='Target')
                #     plt.title(alg_name)
                #     # plt.legend()
                #     plt.tight_layout()
                #     plt.savefig(f'pic/hmm_segmentation/{alg.n_components}_components_player_{test_player}.png')

            # predict = np.array([0] * 18 + list(y_val[:-18]))
                auc_score = scorer(y_val, predict)
                dummy_score = scorer(np.array(y_val), dummy_predict)
            # print(f'{alg.__class__.__name__}:', auc_score)
                auc_scores4alg.append(auc_score)
                dummy_scores.append(dummy_score)

        dummy_score = np.mean(dummy_scores)
        alg_score = np.mean(auc_scores4alg)
        print(f'{alg_name}: {round(alg_score, 3)}')
        print(f'{"Dummy Score"}: {round(dummy_score, 3)}')

        index_array = [[time_step], [window_size], [alg_name], [n_repeat]]
        # index_array = [[5], [120], [2], [8]]

        # = val_score_best
        multi_index = pd.MultiIndex.from_arrays(index_array, names=index_names)

        df_results.loc[multi_index] = alg_score



df_results.to_csv(f'data/df_results_classic_{suffix}.csv')







    # knn = KNeighborsClassifier()
    # knn.fit(x_train, y_train)
    # predict = lr.predict(x_val)
    # auc_score = scorer(y_val, predict)
    # print('KNN:', auc_score)



    # cross_val_score(lr, x_train, y_train, scoring='roc_auc', cv=5)

