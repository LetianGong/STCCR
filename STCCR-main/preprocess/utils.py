from scipy.sparse import csr_matrix
from math import radians, cos, sin, asin, sqrt
import numpy as np
import torch
# import dgl


def construct_spatial_graph_according_to_Events(us, vs, venue_cnt, theta=1):
    '''
    construct spatial graph according to check-in sequences
    构造位置节点矩阵，theta表示阈值，只有两个节点的连接次数超过theta时,才有边
    :param us: source nodes
    :param vs: target nodes
    :param venue_cnt: cnt of spatial nodes
    :param theta:
    :return:
    '''
    print('construct_spatial_graph....')
    uv_weight = {}
    ulist = []
    vlist = []
    for i in range(len(us)):
        k = str(us[i])+'-'+str(vs[i])
        # print(k)
        if k in uv_weight.keys():
            uv_weight[k] = uv_weight[k] + 1
        else:
            uv_weight[k] = 1
            ulist.append(us[i])
            vlist.append(vs[i])
    ulist = np.array(ulist)
    vlist = np.array(vlist)
    weights = list(uv_weight.values())
    weights = list(map(lambda x : 1 if x > theta else 0, weights))
    SS = csr_matrix((weights, (ulist, vlist)), shape=(venue_cnt, venue_cnt)).toarray()
    ss_g = dgl.DGLGraph(SS)
    print('ss_g:', ss_g)
    return ss_g, SS


def distance(lon1, lat1, lon2, lat2):  # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    if lon1 == 0 or lat1 ==0 or lon2==0 or lat2==0:
        return 0
    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    return c * r  # 单位公里


def construct_spatial_matrix_accordingDistance(distance_theta, venue_cnt, venue_lng, venue_lat):
    """
    SS_distance is a matrix records the distance (km) between two locations;
    SS_proximity is a 0-1 matrix, if the distance (km) between two locations <= distance_theta 1, else 0;
    """
    SS_distance = np.zeros((venue_cnt, venue_cnt))
    SS_proximity = np.zeros((venue_cnt, venue_cnt))
    for i in range(venue_cnt):
        for j in range(venue_cnt):
            SS_distance[i, j] = distance(venue_lng[i], venue_lat[i], venue_lng[j], venue_lat[j])
            if SS_distance[i, j] <= distance_theta:
                SS_proximity[i, j] = 1
    return SS_distance, SS_proximity


def construct_spatial_graph_according_to_distance(distance_theta, venue_cnt, venue_lng, venue_lat):
    SS_distance, SS_proximity = construct_spatial_matrix_accordingDistance(distance_theta, venue_cnt, venue_lng, venue_lat)
    SS = csr_matrix(SS_proximity)
    ss_g = dgl.DGLGraph(SS)
    print('ss_g:', ss_g)
    return ss_g, SS_proximity


def construct_temporal_graph(time_interval_minutes):
    print('construct_temporal_graph....')
    time2tid = {}
    num_weekday_embeddings = 7
    num_minute_embeddings = 24 * 60 // time_interval_minutes
    num_temporal_nodes = num_weekday_embeddings * num_minute_embeddings
    print('num_temporal_nodes:', num_temporal_nodes)

    # 时间节点的星期属性特征
    week_info = []
    # 时间节点的时间段索引特征
    time_info = []
    for w in range(num_weekday_embeddings):
        for t in range(num_minute_embeddings):
            week_info.append(w)
            time_info.append(t)

    # week+hour+minute --> nid
    nid = 0
    for i in range(len(week_info)):
        w = week_info[i]
        h, m = time_info[i]//(60 // time_interval_minutes), time_info[i]%(60 // time_interval_minutes)
        # print(str(w)+'-'+str(h)+'-'+str(m))
        time2tid[str(w)+'-'+str(h)+'-'+str(m)] = nid
        nid += 1

    # 构造边：u存储source nodes, v存储target nodes
    u = []
    v = []
    for i in range(num_temporal_nodes):
        for j in range(num_temporal_nodes):
            # proximity: 相邻时间点加边; 自连接;
            if abs(i - j) <= 1 or abs(i - j) == (num_temporal_nodes - 1):
                u.append(i)
                v.append(j)
                # print('proximity:', i, ',', j)
            # periodicity: 相邻天,相同时间点加边;
            if (time_info[i] == time_info[j]) and (
                    (abs(week_info[i] % num_weekday_embeddings - week_info[j] % num_weekday_embeddings) == 1) or (
                    abs(week_info[i] % num_weekday_embeddings - week_info[j] % num_weekday_embeddings) == (
                    num_weekday_embeddings - 1))):
                u.append(i)
                v.append(j)
                # print('periodicity:', i, ',', j)
    u = np.array(u)
    v = np.array(v) #这个u,v列表，已经考虑了双向的关系
    tt_g = dgl.DGLGraph((u, v))
    print('tt_g"', tt_g)
    data = np.ones_like(u)
    TT = csr_matrix((data, (u, v)), shape=(num_temporal_nodes, num_temporal_nodes)).toarray()

    # construct temporal points features
    time_info = torch.from_numpy(np.array(time_info)).long()
    week_info = torch.from_numpy(np.array(week_info)).long()
    return tt_g, time_info, week_info, time2tid, TT


def construct_TS_graph(venue_cnt, temporal_node_cnt, time2tid, trainX_locations, trainX_arrival_datetimes_weekday, trainX_arrival_datetimes_hour, trainX_arrival_datetimes_minute, theta=1):
    # according to time-location in train get ST-relation
    print('construct_TS_graph....')
    uT_vL_weight = {}  # source nodes are temporal nodes, target nodes are spatial/locations nodes
    for i in range(len(trainX_arrival_datetimes_weekday)):
        for j in range(len(trainX_arrival_datetimes_weekday[i])):
            w = trainX_arrival_datetimes_weekday[i][j]
            h = trainX_arrival_datetimes_hour[i][j]
            m = trainX_arrival_datetimes_minute[i][j]
            tid = time2tid[str(w) + '-' + str(h) + '-' + str(m)]
            lid = trainX_locations[i][j]
            k = str(tid) + '-' + str(lid)
            if k in uT_vL_weight.keys():
                uT_vL_weight[k] = uT_vL_weight[k] + 1
            else:
                uT_vL_weight[k] = 1
    uT = []
    vL = []
    for key in uT_vL_weight.keys():
        t, l = key.split('-')
        uT.append(int(t))
        vL.append(int(l))
    uT = np.array(uT)
    vL = np.array(vL)
    data = list(uT_vL_weight.values())
    data = list(map(lambda x:1 if x>theta else 0, data))
    TS = csr_matrix((data, (uT, vL)), shape=(temporal_node_cnt, venue_cnt)).toarray()
    return TS