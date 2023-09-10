import torch.utils.data as data_utils
import os
import itertools
from preprocess.utils import *
# import dgl
from scipy.sparse import csr_matrix

def deal_session_ArrivalTimes(X_session_arrival_datetimes_weekday, X_session_arrival_datetimes_hour, X_session_arrival_datetimes_minute, time_interval_minutes):
    '''
    根据weekday\hour\mins 得到以time_interval_minutes为单位的时间片的index.
    :param session_arrival_times:
    :param time_interval_minutes:
    :return:
    '''
    X_session_arrival_times_index = []

    for i in range(len(X_session_arrival_datetimes_weekday)):
        # 依次处理每个样本，一个sequence即为一个样本，有多个sessions
        sample_timeindex = []

        X_session_arrival_datetimes_minute_i = [list(map(lambda x: int(x / time_interval_minutes), times)) for times in X_session_arrival_datetimes_minute[i]]

        for j in range(len(X_session_arrival_datetimes_weekday[i])):  # 依次处理一个sequence样本的多个sessions

            times_index = [24 * 60 // time_interval_minutes * week + 60 // time_interval_minutes * hour + mins for
                           week, hour, mins in zip(X_session_arrival_datetimes_weekday[i][j], X_session_arrival_datetimes_hour[i][j], X_session_arrival_datetimes_minute_i[j])]
            sample_timeindex.append(times_index)

        X_session_arrival_times_index.append(sample_timeindex)

    return X_session_arrival_times_index



def load_data_for_STCCR_TP(name, data_root, save_split, time_interval_minutes, proximity_T=False, proximity_S=False, semantic_T=False, semantic_S=False, thetaSS=1, thetaTS_T=1, thetaTS_S=100, localGPU=False):
    '''
    1. load data and construct train/val/test dataset
    2. construct temporal graphs gts and spatial graphs gss
    3. construct SessionBasedSequenceDataset
    :param name: file name
    :param name: save_split, whether train/val/test samples are saved in separated files
    :return:
    '''
    gts = []
    gss = []
    time_info = None
    week_info = None
    max_len_x = 0
    min_len_x = 10000

    if not name.endswith('.npz'):
        name += '.npz'

    if localGPU:
        if save_split:
            train_loader = dict(np.load(os.path.join(data_root, 'train_' + name), allow_pickle=True))
            val_loader = dict(np.load(os.path.join(data_root , 'val_' + name), allow_pickle=True))
            loader = dict(np.load(os.path.join(data_root , 'test_' + name), allow_pickle=True))
        else:
            loader = dict(np.load(os.path.join(data_root, 'test_' + name), allow_pickle=True))
            train_loader = loader
            val_loader = loader

    user_cnt = loader['user_cnt']
    venue_cnt = loader['venue_cnt']
    print('user_cnt:', user_cnt)
    print('venue_cnt:', venue_cnt)

    if proximity_T or proximity_S or semantic_T or semantic_S:
        # construct temporal graph and temporal point features
        tt_g, time_info, week_info, time2tid, TT = construct_temporal_graph(time_interval_minutes)
    distance_matrix = loader['SS_distance']
    guassian_distance_matrix = loader['SS_guassian_distance']
    print('distance_matrix:', distance_matrix.shape, 'guassian_distance_matrix:', guassian_distance_matrix.shape)

    # user_lidfreq = torch.Tensor(train_loader['user_lidfreq'])
    # print('user_lidfreq:', user_lidfreq.shape)
    # construct spatial graph
    # ----- load spatial graph info -----
    # us = loader['us']
    # vs = loader['vs']
    feature_lat = loader['feature_lat']  # index
    feature_lng = loader['feature_lng']  # index
    feature_lat_ori = loader['feature_lat_ori']  # numerical 记录真实的经纬度信息 (Ns,)
    feature_lng_ori = loader['feature_lng_ori']  # numerical (Ns,)
    feature_lat_ori = torch.Tensor(feature_lat_ori)
    feature_lng_ori = torch.Tensor(feature_lng_ori)
    print('feature_lng_ori:', feature_lng_ori.shape)

    if proximity_T or proximity_S or semantic_T or semantic_S:

        ## 根据经纬度距离，构造SS，经纬度距离在thetaSS范围内的点连边
        ss_g, SS = construct_spatial_graph_according_to_distance(thetaSS, venue_cnt, feature_lng_ori, feature_lat_ori)

    # put spatial point features into tensor
    feature_lat = torch.LongTensor(feature_lat)
    feature_lng = torch.LongTensor(feature_lng)

    latN, lngN = loader['latN'], loader['lngN']
    category_cnt = loader['category_cnt']
    print('category_cnt:', category_cnt)

    if category_cnt > 0:
        feature_category = loader['feature_category']
        feature_category = torch.LongTensor(feature_category)
        print('feature_category:', feature_category.shape)
    else:
        feature_category = None

    # ----- load train, turn to [numpy],  get train_dataset -----
    trainX_arrival_datetimes_weekday = train_loader['trainX_local_weekdays']
    trainX_arrival_datetimes_hour = train_loader['trainX_local_hours']
    trainX_arrival_datetimes_minute = train_loader['trainX_local_mins']
    trainX_duration2first = train_loader['trainX_duration2first']
    trainX_locations = train_loader['trainX_locations']
    trainX_tau = pad_1D(train_loader['trainX_delta_times'])
    trainX_users = train_loader['trainX_users']
    trainX_lengths = train_loader['trainX_lengths']
    if min(trainX_lengths) <  min_len_x:
        min_len_x = min(trainX_lengths)
    if max(trainX_lengths) > max_len_x:
        max_len_x = max(trainX_lengths)
    trainX_session_lengths = train_loader['trainX_session_lengths']
    trainX_session_num = train_loader['trainX_session_num']
    trainY_delta_times = pad_1D(train_loader['trainY_delta_times'])
    trainY_locations =  pad_1D(train_loader['trainY_locations'])

    trainX_arrival_datetimes_minute = [list(map(lambda x: int(x / time_interval_minutes), times)) for times in
                                       trainX_arrival_datetimes_minute]

    trainX_arrival_times_index = [list(
        map(lambda w, h, m: 24 * 60 // time_interval_minutes * w + 60 // time_interval_minutes * h + m, week, hour,
            mins)) for week, hour, mins in
        zip(trainX_arrival_datetimes_weekday, trainX_arrival_datetimes_hour, trainX_arrival_datetimes_minute)]

    # trainX_arrival_timeinterval_index 与 trainX_arrival_times_index 的区别在于，上面区分星期，下面不区分星期；下面与Tembed中输入的节点的index特征一致

    trainX_arrival_timeinterval_index = [list(map(lambda h, m: 60 // time_interval_minutes * h + m, hour, mins)) for hour, mins in zip(trainX_arrival_datetimes_hour, trainX_arrival_datetimes_minute)]

    if proximity_T or proximity_S or semantic_T or semantic_S:
        TS_t = construct_TS_graph(venue_cnt, len(time_info), time2tid, trainX_locations,
                                  trainX_arrival_datetimes_weekday, trainX_arrival_datetimes_hour,
                                  trainX_arrival_datetimes_minute, theta=thetaTS_T)

        TS_s = construct_TS_graph(venue_cnt, len(time_info), time2tid, trainX_locations,
                                  trainX_arrival_datetimes_weekday, trainX_arrival_datetimes_hour,
                                  trainX_arrival_datetimes_minute, theta=thetaTS_S)

        TST = np.matmul(TS_t, np.transpose(TS_t))
        STS = np.matmul(np.transpose(TS_s), TS_s)

        if proximity_T:
            gts.append(tt_g)
        if semantic_T:
            tst_g = dgl.DGLGraph(csr_matrix(TST))
            tst_g = dgl.add_self_loop(tst_g)
            print('tst_g:', tst_g)
            gts.append(tst_g)
        if proximity_S:
            gss.append(ss_g)
        if semantic_S:
            sts_g = dgl.DGLGraph(csr_matrix(STS))
            sts_g = dgl.add_self_loop(sts_g)
            print('sts_g:', sts_g)
            gss.append(sts_g)

    # turn data with variable length into numpy
    trainX_arrival_datetimes_weekday = [np.concatenate([x]) for x in
                                        trainX_arrival_datetimes_weekday]
    trainX_arrival_datetimes_hour = [np.concatenate([x]) for x in trainX_arrival_datetimes_hour]
    trainX_arrival_datetimes_minute = [np.concatenate([x]) for x in trainX_arrival_datetimes_minute]
    trainX_arrival_times_index = [np.concatenate([x]) for x in trainX_arrival_times_index]
    trainX_arrival_timeinterval_index = [np.concatenate([x]) for x in trainX_arrival_timeinterval_index]

    data_train = SessionBasedSequenceDataset(user_cnt, venue_cnt, trainX_lengths,
                                             trainX_arrival_datetimes_weekday, trainX_arrival_datetimes_hour, trainX_arrival_datetimes_minute, trainX_arrival_times_index, trainX_arrival_timeinterval_index,
                                             trainX_locations,
                                             trainX_users, trainX_session_lengths, trainX_session_num, trainY_delta_times, trainY_locations, trainX_duration2first, trainX_tau)

    print('samples cnt of data_train:', data_train.num_series)
    # return data_train, data_train, data_train, gts, gss, time_info, week_info, feature_category, feature_lat, feature_lng, latN, lngN, category_cnt, feature_lat_ori, feature_lng_ori

    # ----- load val, turn to [numpy],  get val_dataset -----
    valX_arrival_datetimes_weekday = val_loader['valX_local_weekdays']
    valX_arrival_datetimes_hour = val_loader['valX_local_hours']
    valX_arrival_datetimes_minute = val_loader['valX_local_mins']
    valX_duration2first = val_loader['valX_duration2first']
    valX_tau = pad_1D(val_loader['valX_delta_times'])
    valX_locations = val_loader['valX_locations']
    valX_users = val_loader['valX_users']
    valX_lengths = val_loader['valX_lengths']
    if min(valX_lengths) <  min_len_x:
        min_len_x = min(valX_lengths)
    if max(valX_lengths) > max_len_x:
        max_len_x = max(valX_lengths)

    valX_session_lengths = val_loader['valX_session_lengths']
    valX_session_num = val_loader['valX_session_num']
    valY_delta_times = pad_1D(val_loader['valY_delta_times'])
    valY_locations =  pad_1D(val_loader['valY_locations'])

    valX_arrival_datetimes_minute = [list(map(lambda x: int(x / time_interval_minutes), times)) for times in
                                       valX_arrival_datetimes_minute]

    valX_arrival_times_index = [list(
        map(lambda w, h, m: 24 * 60 // time_interval_minutes * w + 60 // time_interval_minutes * h + m, week, hour,
            mins)) for week, hour, mins in
        zip(valX_arrival_datetimes_weekday, valX_arrival_datetimes_hour, valX_arrival_datetimes_minute)]

    valX_arrival_timeinterval_index = [list(map(lambda h, m: 60 // time_interval_minutes * h + m, hour, mins)) for
                                         hour, mins in
                                         zip(valX_arrival_datetimes_hour, valX_arrival_datetimes_minute)]

    # turn data with variable length into numpy
    valX_arrival_datetimes_weekday = [np.concatenate([x]) for x in
                                        valX_arrival_datetimes_weekday]
    valX_arrival_datetimes_hour = [np.concatenate([x]) for x in
                                        valX_arrival_datetimes_hour]
    valX_arrival_datetimes_minute = [np.concatenate([x]) for x in
                                        valX_arrival_datetimes_minute]
    valX_arrival_times_index = [np.concatenate([x]) for x in valX_arrival_times_index]
    valX_arrival_timeinterval_index = [np.concatenate([x]) for x in valX_arrival_timeinterval_index]

    data_val = SessionBasedSequenceDataset(user_cnt, venue_cnt, valX_lengths,
                                             valX_arrival_datetimes_weekday, valX_arrival_datetimes_hour, valX_arrival_datetimes_minute, valX_arrival_times_index, valX_arrival_timeinterval_index,
                                             valX_locations,
                                             valX_users, valX_session_lengths, valX_session_num,
                                           valY_delta_times, valY_locations, valX_duration2first, valX_tau)

    print('samples cnt of data_val:', data_val.num_series)

    # ----- load test, turn to [numpy],  get test_dataset -----
    testX_arrival_datetimes_weekday = loader['testX_local_weekdays']
    testX_arrival_datetimes_hour = loader['testX_local_hours']
    testX_arrival_datetimes_minute = loader['testX_local_mins']
    testX_duration2first = loader['testX_duration2first']
    testX_tau = pad_1D(loader['testX_delta_times'])
    testX_locations = loader['testX_locations']
    testX_users = loader['testX_users']
    testX_lengths = loader['testX_lengths']
    if min(testX_lengths) <  min_len_x:
        min_len_x = min(testX_lengths)
    if max(testX_lengths) > max_len_x:
        max_len_x = max(testX_lengths)
    testX_session_lengths = loader['testX_session_lengths']
    testX_session_num = loader['testX_session_num']
    testY_delta_times = pad_1D(loader['testY_delta_times'])
    testY_locations =  pad_1D(loader['testY_locations'])

    testX_arrival_datetimes_minute = [list(map(lambda x: int(x / time_interval_minutes), times)) for times in
                                       testX_arrival_datetimes_minute]

    testX_arrival_times_index = [list(
        map(lambda w, h, m: 24 * 60 // time_interval_minutes * w + 60 // time_interval_minutes * h + m, week, hour,
            mins)) for week, hour, mins in
        zip(testX_arrival_datetimes_weekday, testX_arrival_datetimes_hour, testX_arrival_datetimes_minute)]

    testX_arrival_timeinterval_index = [list(map(lambda h, m: 60 // time_interval_minutes * h + m, hour, mins)) for
                                         hour, mins in
                                         zip(testX_arrival_datetimes_hour, testX_arrival_datetimes_minute)]

    # turn data with variable length into numpy
    testX_arrival_datetimes_weekday = [np.concatenate([x]) for x in
                                      testX_arrival_datetimes_weekday]
    testX_arrival_datetimes_hour = [np.concatenate([x]) for x in testX_arrival_datetimes_hour]
    testX_arrival_datetimes_minute = [np.concatenate([x]) for x in testX_arrival_datetimes_minute]
    testX_arrival_times_index = [np.concatenate([x]) for x in testX_arrival_times_index]
    testX_arrival_timeinterval_index = [np.concatenate([x]) for x in testX_arrival_timeinterval_index]

    data_test = SessionBasedSequenceDataset(user_cnt, venue_cnt, testX_lengths,
                                           testX_arrival_datetimes_weekday, testX_arrival_datetimes_hour, testX_arrival_datetimes_minute, testX_arrival_times_index, testX_arrival_timeinterval_index,
                                           testX_locations,
                                           testX_users, testX_session_lengths, testX_session_num, testY_delta_times, testY_locations, testX_duration2first, testX_tau)

    print('samples cnt of data_test:', data_test.num_series)
    user_lidfreq = None
    return data_train, data_val, data_test, gts, gss, time_info, week_info, feature_category, feature_lat, feature_lng, latN, lngN, category_cnt, feature_lat_ori, feature_lng_ori, distance_matrix, guassian_distance_matrix, user_lidfreq, min_len_x, max_len_x

class SessionBasedSequenceDataset(data_utils.Dataset):
    """Dataset class containing variable length sequences.
    """
    def __init__(self, user_cnt, venue_cnt, X_lengths,
                 X_arrival_datetimes_weekday, X_arrival_datetimes_hour, X_arrival_datetimes_minute, X_arrival_times_index, X_arrival_timeinterval_index, X_locations,
                 X_users, X_session_lengths, X_session_num, Y_taus, Y_locations, X_duration2first, X_taus):

        self.user_cnt = user_cnt
        self.venue_cnt = venue_cnt
        self.X_lengths = torch.Tensor(X_lengths)
        self.X_arrival_datetimes_weekday = [torch.Tensor(_) for _ in X_arrival_datetimes_weekday]
        self.X_arrival_datetimes_hour = [torch.Tensor(_) for _ in X_arrival_datetimes_hour]
        self.X_arrival_datetimes_minute = [torch.Tensor(_) for _ in X_arrival_datetimes_minute]
        self.X_arrival_times_index = [torch.Tensor(_) for _ in X_arrival_times_index]
        self.X_arrival_timeinterval_index = [torch.Tensor(_) for _ in X_arrival_timeinterval_index]
        self.X_taus = [torch.Tensor(_) for _ in X_taus]
        self.X_duration2first = [torch.Tensor(_) for _ in X_duration2first]
        self.X_locations = [torch.Tensor(_) for _ in X_locations]
        # self.X_last_distances = torch.Tensor(X_last_distances)
        self.X_users = torch.Tensor(X_users)
        # self.Y_taus = torch.Tensor(Y_taus.astype('float64'))/60  # mins->hour
        self.Y_taus = torch.Tensor(Y_taus.astype('float64'))  # mins->hour
        self.Y_locations = torch.Tensor(Y_locations)
        # print('self.X_locations, self.Y_locations:')
        # print(self.X_locations[0])
        # print(self.Y_locations[0])
        # self.aux_y = [torch.cat((x[1:], y.unsqueeze(-1)), -1)for x, y in zip(self.X_locations, self.Y_locations)]
        # print('self.aux_y:', self.aux_y[0])
        self.X_session_num = torch.Tensor(X_session_num)
        self.X_session_lengths = [torch.Tensor(_) for _ in X_session_lengths]

        # self.X_session_arrival_datetimes_weekday = X_session_arrival_datetimes_weekday
        # self.X_session_arrival_datetimes_hour = X_session_arrival_datetimes_hour
        # self.X_session_arrival_datetimes_minute = X_session_arrival_datetimes_minute
        # self.X_session_arrival_times_index = X_session_arrival_times_index
        # self.X_session_taus = X_session_taus
        # self.X_session_locations = X_session_locations

        self.validate_data()

    @property
    def num_series(self):
        return len(self.X_locations)  # num of sequence samples

    def validate_data(self):
        if len(self.X_users) != len(self.X_lengths) or len(self.X_users) != len(self.Y_taus) or len(
                self.X_users) != len(self.Y_locations):
            raise ValueError("Length of X_users, X_lengths, Y_taus, Y_locations should match")
        if len(self.X_users) != len(self.X_arrival_datetimes_weekday):
            raise ValueError("Length of X_users, X_arrival_datetimes_weekday should match")

        for s1, s2, s3 in zip(self.X_arrival_datetimes_weekday, self.X_arrival_times_index, self.X_locations):
            if len(s1) != len(s2) or len(s1) != len(s3):
                raise ValueError("Some input arrival series have different lengths.")

        # if len(self.X_users) != len(self.X_arrival_datetimes_weekday) or len(self.X_users) != len(self.X_taus):
        #     raise ValueError("Length of X_users, X_arrival_datetimes_weekday, X_taus should match")
        #
        # for s1, s2, s3, s4, s5, s6 in zip(self.X_arrival_datetimes_weekday, self.X_arrival_datetimes_hour,
        #                                   self.X_arrival_datetimes_minute, self.X_arrival_times_index, self.X_taus,
        #                                   self.X_locations):
        #     if len(s1) != len(s2) or len(s1) != len(s3) or len(s1) != len(s4) or len(s1) != len(s5) or len(s1) != len(
        #             s6):
        #         raise ValueError("Some input arrival series have different lengths.")
    #todo 生成mean和std
    def get_tau_log_mean_std_Y(self):
        """Get mean and std of Y_taus."""
        y = torch.flatten(self.Y_taus)
        logy = y[y != 0].log()
        return logy.mean(), logy.std()

    def get_mean_std_Y_tau(self):
        """Get mean and std of Y_tau."""
        y = torch.flatten(self.Y_taus)
        y = y[y != 0]
        return y.mean(), y.std()

    def normalize_Y_tau(self, mean=None, std=None):
        self.Y_taus = (self.Y_taus - mean)/std
        return self

    def __getitem__(self, key):
        '''
        the outputs are feed into collate()
        :param key:
        :return:
        '''
        return self.X_arrival_datetimes_weekday[key], self.X_arrival_datetimes_hour[key], \
               self.X_arrival_datetimes_minute[key], self.X_arrival_times_index[key], \
               None, None, \
               None, None, \
               self.X_taus[key], None, self.X_locations[key], None, \
               None, self.X_users[key], self.X_session_num[key], self.X_session_lengths[key], \
               self.Y_taus[key], self.Y_locations[key], self.X_lengths[key], None, self.X_duration2first[
                   key], self.X_arrival_timeinterval_index[key]

        # return self.X_arrival_datetimes_weekday[key], self.X_arrival_datetimes_hour[key], self.X_arrival_datetimes_minute[key], self.X_arrival_times_index[key], \
        #        self.X_session_arrival_datetimes_weekday[key], self.X_session_arrival_datetimes_hour[key], self.X_session_arrival_datetimes_minute[key], self.X_session_arrival_times_index[key], \
        #        self.X_taus[key], self.X_session_taus[key], self.X_locations[key], self.X_session_locations[key], self.X_last_distances[key], self.X_users[key], self.X_session_num[key], self.X_session_lengths[key], \
        #        self.Y_taus[key], self.Y_locations[key], self.X_lengths[key], self.aux_y[key], self.X_duration2first[key], self.X_arrival_timeinterval_index[key]

    def __len__(self):
        return self.num_series

    def __repr__(self):
        # return f"SequenceDataset({self.num_series})" py3.6

        return "SequenceDataset" + self.num_series


def pad_session_data(X_session_data1, X_session_data2, X_session_data3, X_session_lengths):
    '''

    :param X_session_data1:
    :param X_session_data2:
    :param X_session_data3:
    :param X_session_lengths: # (batch, max_length)
    :return: all_samples_1, all_samples_2, all_samples_3： （batch,）
    '''
    max_session_length = int(max([max(_) for _ in X_session_lengths]))  # 找到这个batch中最长的session
    fillvalue = max_session_length*[0]

    # 将所有sample都处理成等session num
    data1_padded_to_same_session_num = list(zip(*itertools.zip_longest(*X_session_data1, fillvalue=fillvalue)))  # 缺少的session都填充为最长长度的session
    data2_padded_to_same_session_num = list(zip(*itertools.zip_longest(*X_session_data2, fillvalue=fillvalue)))
    data3_padded_to_same_session_num = list(zip(*itertools.zip_longest(*X_session_data3, fillvalue=fillvalue)))

    # 将所有session padding成最大长度
    all_samples_1 = []
    all_samples_2 = []
    all_samples_3 = []

    for i in range(len(data1_padded_to_same_session_num)):  # 遍历第i个session
        sample_1 = data1_padded_to_same_session_num[i]
        padded_sample_1 = list(zip(* itertools.zip_longest(*sample_1, fillvalue=0)))
        padded_sample_1 = [list(x) for x in padded_sample_1]

        sample_2 = data2_padded_to_same_session_num[i]
        padded_sample_2 = list(zip(*itertools.zip_longest(*sample_2, fillvalue=0)))
        padded_sample_2 = [list(x) for x in padded_sample_2]

        sample_3 = data3_padded_to_same_session_num[i]
        padded_sample_3 = list(zip(*itertools.zip_longest(*sample_3, fillvalue=0)))
        padded_sample_3 = [list(x) for x in padded_sample_3]

        if len(padded_sample_1[0]) < max_session_length:
            padding = (max_session_length-len(padded_sample_1[0]))*[0]
            for session_1 in padded_sample_1:
                session_1.extend(padding)
            for session_2 in padded_sample_2:
                session_2.extend(padding)
            for session_3 in padded_sample_3:
                session_3.extend(padding)

        all_samples_1.append(padded_sample_1)
        all_samples_2.append(padded_sample_2)
        all_samples_3.append(padded_sample_3)

    return all_samples_1, all_samples_2, all_samples_3


def generate_session_mask(length, session_num, session_lengths):
    '''

    :param length: list, len(length)=batch size
    :param session_num:
    :param session_lengths:
    :return: bool matrix with size (batch, max_length, max_length), True (1) means no attention
    '''
    # print('length:', type(length), length)
    # print('session_num:', type(session_num), session_num)
    # print('session_lengths:', type(session_lengths), session_lengths)
    max_length = int(length[0])
    batch_size = len(length)
    mask = torch.ones(batch_size, max_length, max_length).type(torch.uint8)  # (B, max_length, max_length)
    p = [0] * batch_size

    for i in range(batch_size):
        for j in range(len(session_lengths[i])):
            current_session_length = int(session_lengths[i][j].cpu())
            current_session_mask = torch.zeros(current_session_length, current_session_length).type(torch.uint8)
            # print('i=', i, 'j=', j, 'current_session_mask:', current_session_mask)
            mask[i, p[i]:(p[i] + current_session_length), p[i]:(p[i] + current_session_length)] = current_session_mask
            p[i] += current_session_length

    return mask


def collate_session_based(batch):
    '''
    get the output of dataset.__getitem__, and perform padding
    :param batch:
    :return:
    '''
    # device = batch[0][-1]

    # generate_session_mask(X_length, X_session_num, X_session_lengths)

    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)  # 按着check-in序列个数从大到小进行排序

    X_weekday_idx = [item[0] for item in batch]
    X_hour_idx = [item[1] for item in batch]
    X_min_idx = [item[2] for item in batch]

    X_timeidx = [item[3] for item in batch]
    target_lengths = [item[4] for item in batch]
    # X_session_arrival_times_index = [item[7] for item in batch]

    X_tau = torch.stack([item[8] for item in batch])

    torch.where(X_tau < 5, torch.mean(X_tau), X_tau)
    # X_session_tau = [item[9] for item in batch]

    X_location = [item[10] for item in batch]
    # print('X_location:', X_location)

    # X_session_locations = [item[11] for item in batch]
    #
    # X_last_distance = torch.stack([item[12] for item in batch])

    X_user = torch.tensor([item[13] for item in batch])

    X_session_num = [item[14] for item in batch]

    X_session_lengths = [item[15] for item in batch]

    Y_tau = torch.stack([item[16] for item in batch])

    torch.where(Y_tau < 5, torch.mean(Y_tau), Y_tau)

    Y_location = torch.stack([item[17] for item in batch])

    X_length = [item[18] for item in batch]

    session_mask = generate_session_mask(X_length, X_session_num, X_session_lengths)

    # print('Y_location:', Y_location[:2])

    # aux_y = [item[19] for item in batch]
    # print('aux_y:', aux_y[:2])

    X_duration21 = [item[20] for item in batch]
    X_timeinterval_idx = [item[21] for item in batch]

    X_length = torch.tensor(X_length)  # to tensor

    # X_session_num = torch.tensor(X_session_num) # to tensor

    X_timeidx = list(zip(*(itertools.zip_longest(*X_timeidx, fillvalue=0))))
    X_timeidx = torch.tensor(X_timeidx)  # (batch, max_length)

    X_timeinterval_idx = list(zip(*(itertools.zip_longest(*X_timeinterval_idx, fillvalue=0))))
    X_timeinterval_idx = torch.tensor(X_timeinterval_idx)  # (batch, max_length)

    # print("X_weekday_idx: ", X_weekday_idx)
    X_weekday_idx = list(zip(*(itertools.zip_longest(*X_weekday_idx, fillvalue=0))))
    X_weekday_idx = torch.tensor(X_weekday_idx)  # (batch, max_length)
    # print("X_hour_idx: ", X_hour_idx)
    X_hour_idx = list(zip(*(itertools.zip_longest(*X_hour_idx, fillvalue=0))))
    X_hour_idx = torch.tensor(X_hour_idx)  # (batch, max_length)
    X_min_idx = list(zip(*(itertools.zip_longest(*X_min_idx, fillvalue=0))))
    X_min_idx = torch.tensor(X_min_idx)  # (batch, max_length)

    X_tau = list(zip(*(itertools.zip_longest(*X_tau, fillvalue=0))))
    X_tau = torch.tensor(X_tau)  # (batch, max_length)
    X_duration21 = list(zip(*(itertools.zip_longest(*X_duration21, fillvalue=0))))
    X_duration21 = torch.tensor(X_duration21)  # (batch, max_length)
    X_location = list(zip(*(itertools.zip_longest(*X_location, fillvalue=0))))
    X_location = torch.tensor(X_location)  # (batch, max_length)
    # aux_y = list(zip(*(itertools.zip_longest(*aux_y, fillvalue=0))))
    # aux_y = torch.tensor(aux_y)  # (batch, max_length)
    # X_session_lengths = list(zip(*(itertools.zip_longest(*X_session_lengths, fillvalue=0))))
    # X_session_lengths = torch.tensor(X_session_lengths)  # (batch, max_length)

    # padded_X_session_locations, padded_X_session_arrival_times_index, padded_X_session_tau = pad_session_data(X_session_locations, X_session_arrival_times_index, X_session_tau, X_session_lengths)
    # padded_X_session_locations = torch.tensor(padded_X_session_locations)
    # padded_X_session_arrival_times_index = torch.tensor(padded_X_session_arrival_times_index)
    # padded_X_session_tau = torch.tensor(padded_X_session_tau)
    # return session_Batch(X_weekday_idx, X_timeinterval_idx, X_timeidx, padded_X_session_arrival_times_index, X_tau, X_duration21, padded_X_session_tau, X_location, padded_X_session_locations,
    #                      X_last_distance, X_user, X_session_num, X_session_lengths, Y_tau, Y_location, X_length, session_mask, aux_y)

    return session_Batch(X_weekday_idx, X_hour_idx, X_min_idx, X_timeinterval_idx, X_timeidx, target_lengths, None, X_tau,
                         X_duration21, None, X_location, None,
                         None, X_user, None, None, Y_tau, Y_location, X_length,
                         session_mask, None)


class session_Batch():
    def __init__(self, X_weekday_idx, X_hour_idx, X_min_idx, X_timeinterval_idx, X_timeidx, target_lengths, padded_X_session_arrival_times_index, X_tau, X_duration21, padded_X_session_tau, X_location, padded_X_session_locations, X_last_distance, X_user, X_session_num, X_session_lengths, Y_tau, Y_location, X_length, X_session_mask, aux_y):
        self.X_weekday_idx = X_weekday_idx.long().cuda()#.to(device)
        self.X_hour_idx = X_hour_idx.long().cuda()
        self.X_min_idx = X_min_idx.long().cuda()
        self.X_timeinterval_idx = X_timeinterval_idx.long().cuda()#.to(device)  # (batch, max_length)
        self.X_timeidx = X_timeidx.long().cuda()#.to(device) # (batch, max_length)
        # self.X_session_arrival_times_index = padded_X_session_arrival_times_index.long().cuda()#.to(device) # (batch, max_session_num, max_session_length)
        self.X_tau = X_tau.cuda()#.to(device) # (batch, max_length)
        self.X_duration21 = X_duration21.cuda()#.to(device) # (batch, max_length)
        # self.X_session_tau = padded_X_session_tau.cuda()#.to(device) # (batch, max_session_num, max_session_length)
        self.X_location = X_location.long().cuda()#.to(device)  # (batch, max_length)
        # self.X_session_locations = padded_X_session_locations.long().cuda()#.to(device) # (batch, max_session_num, max_session_length)
        # self.X_last_distance = X_last_distance.cuda()#.to(device) # (batch, num_classes)
        self.X_user = X_user.long().cuda()#.to(device)  # (batch,)
        # self.X_session_num = X_session_num.cuda()#.to(device)  # (batch,)
        # self.X_session_lengths = X_session_lengths.cuda()#.to(device) # (batch, max_session_num)
        self.Y_tau = Y_tau.cuda()#.to(device)# (batch,)
        self.Y_location = Y_location.long().cuda()#.to(device)# (batch,)=(64,)
        self.target_lengths = target_lengths
        self.X_length = X_length.long().cuda()#.to(device) # (batch,)
        self.X_session_mask = X_session_mask.cuda()#.to(device)# (batch, max_length, max_length)
        # self.aux_y = aux_y.long().cuda()#.to(device)

        # print('self.X_timeidx:', self.X_timeidx)
        # print('self.X_session_arrival_times_index:', self.X_session_arrival_times_index)
        # print('self.X_tau:', self.X_tau)
        # print('X_duration21:', self.X_duration21)
        # print('self.X_session_tau:', self.X_session_tau)
        # print('self.X_location:', self.X_location)
        # print('self.X_session_locations:', self.X_session_locations)
        # print('self.X_last_distance:', self.X_last_distance)
        # print('self.X_user:', self.X_user)
        # torch.set_printoptions(threshold=1000000)
        # print('self.X_session_num:', self.X_session_num)
        # print('self.X_session_lengths:', self.X_session_lengths)
        # print('self.Y_tau:', self.Y_tau)
        # print('self.Y_location:', self.Y_location)
        # print('self.X_length:', self.X_length)
        # print('self.X_session_mask:', self.X_session_mask)
        # print('aux_y:', aux_y)


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - len(x)), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])
    return padded
