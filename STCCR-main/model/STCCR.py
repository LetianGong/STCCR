import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from utils import DotDict
from model.utils import *
import torch.nn.functional as F
from torch import nn

class STCCR_ModelConfig(DotDict):
    '''
    configuration of the STCCR
    '''

    def __init__(self, loc_size=None, tim_size=None, uid_size=None, geohash_size=None, category_size=None, tim_emb_size=None, loc_emb_size=None,
                 hidden_size=None, user_emb_size=None, device=None,
                 loc_noise_mean=None, loc_noise_sigma=None, tim_noise_mean=None, tim_noise_sigma=None,
                 user_noise_mean=None, user_noise_sigma=None, tau=None,
                 pos_eps=None, neg_eps=None, dropout_rate_1=None, dropout_rate_2=None, category_vector=None, rnn_type='BiLSTM',
                 num_layers=3, k=8, momentum=0.95, temperature=0.1, theta=0.18,
                 n_components=4, hypernet_hidden_sizes=None, max_delta_mins=1440,
                 downstream='POI', dropout_spatial = None, epsilon = None ):
        super().__init__()
        self.max_delta_mins = max_delta_mins

        self.loc_size = loc_size  #
        self.uid_size = uid_size  # 
        self.tim_size = tim_size  # 
        self.geohash_size = geohash_size  # 
        self.category_size = category_size  # 
        self.loc_emb_size = loc_emb_size
        self.tim_emb_size = tim_emb_size
        self.user_emb_size = user_emb_size
        self.hidden_size = hidden_size  # RNN hidden_size
        self.device = device
        self.rnn_type = rnn_type
        self.num_layers = num_layers

        self.loc_noise_mean = loc_noise_mean
        self.loc_noise_sigma = loc_noise_sigma
        self.tim_noise_mean = tim_noise_mean
        self.tim_noise_sigma = tim_noise_sigma
        self.user_noise_mean = user_noise_mean
        self.user_noise_sigma = user_noise_sigma
        self.tau = tau
        self.pos_eps = pos_eps
        self.neg_eps = neg_eps
        self.dropout_rate_1 = dropout_rate_1
        self.dropout_rate_2 = dropout_rate_2
        self.downstream = downstream
        self.category_vector = category_vector

        self.k = k
        self.momentum = momentum
        self.theta = theta
        self.temperature = temperature

        self.n_components = n_components   
        self.hypernet_hidden_sizes = hypernet_hidden_sizes  
        self.decoder_input_size = user_emb_size + hidden_size * 2   
        self.dropout_spatial = dropout_spatial
        self.epsilon = epsilon

class STCCR(nn.Module):
    def __init__(self, config):
        super(STCCR, self).__init__()
        # initialize parameters
        self.max_delta_mins = config['max_delta_mins']
        self.truth_Y_tau = None
        self.loc_size = config['loc_size']
        self.loc_emb_size = config['loc_emb_size']
        self.tim_size = config['tim_size']
        self.tim_emb_size = config['tim_emb_size']
        self.user_size = config['uid_size']
        self.user_emb_size = config['user_emb_size']

        self.category_size = config['category_size']
        self.geohash_size = config['geohash_size']
        self.category_vector = config['category_vector']

        self.hidden_size = config['hidden_size']
        self.rnn_type = config['rnn_type']
        self.num_layers = config['num_layers']
        self.device = config['device']
        self.downstream = config['downstream']

        # parameters for cluster contrastive learning
        self.k = config['k']

        # parameters for time contrastive learning (Angle & Momentum based)
        # momentum
        self.momentum = config['momentum']
        # angle
        self.theta = config['theta']
        self.temperature = config['temperature']

        # spatial 
        self.user_centroids = None
        self.user_2cluster = None
        self.item_centroids = None
        self.item_2cluster = None
        self.softmax = nn.Softmax()
        self.epsilon = config['epsilon']
        self.sinkhorn_iterations = 3
        self.crops_for_assign = [0, 1]
        self.nmb_crops = [2]
        self.world_size = -1
        self.dropout = nn.Dropout(config['dropout_spatial'])
        self.l2norm = True
        # location all size (embedding + geohash + category)
        self.rnn_input_size = self.loc_emb_size + self.geohash_size + self.category_size

        if self.rnn_type == 'BiLSTM':
            self.bi = 2
        else:
            self.bi = 1

        # parameters for social contrastive learning (4 group of parameters)
        self.para0 = nn.Parameter(torch.randn(1, 6))
        self.para1 = nn.Parameter(torch.randn(1, 4))
        self.para2 = nn.Parameter(torch.randn(1, 24))
        self.para3 = nn.Parameter(torch.randn(1, 16))

        ##############################################
        self.loc_noise_mean = config['loc_noise_mean']
        self.loc_noise_sigma = config['loc_noise_sigma']
        self.tim_noise_mean = config['tim_noise_mean']
        self.tim_noise_sigma = config['tim_noise_sigma']
        self.user_noise_mean = config['user_noise_mean']
        self.user_noise_sigma = config['user_noise_sigma']

        self.tau = config['tau']
        self.pos_eps = config['pos_eps']
        self.neg_eps = config['neg_eps']
        self.dropout_rate_1 = config['dropout_rate_1']
        self.dropout_rate_2 = config['dropout_rate_2']

        self.dropout_1 = nn.Dropout(self.dropout_rate_1)
        self.dropout_2 = nn.Dropout(self.dropout_rate_2)
        ################################################

        # Embedding layer
        self.emb_loc = nn.Embedding(num_embeddings=self.loc_size, embedding_dim=self.loc_emb_size)
        self.emb_tim = nn.Embedding(num_embeddings=self.tim_size, embedding_dim=self.tim_emb_size)
        self.emb_user = nn.Embedding(num_embeddings=self.user_size, embedding_dim=self.user_emb_size)

        # Category dense layer
        self.category_dense = nn.Linear(768, self.category_size)
        # Geohash dense layer
        self.geohash_dense = nn.Linear(12, self.geohash_size)

        # rnn layer
        if self.rnn_type == 'GRU':
            self.spatial_encoder = nn.GRU(self.rnn_input_size, self.hidden_size, num_layers=self.num_layers,
                               batch_first=False)
            self.temporal_encoder = nn.GRU(self.tim_emb_size + 1, self.hidden_size, num_layers=self.num_layers,
                               batch_first=False)
            self.temporal_encoder_momentum = nn.GRU(self.tim_emb_size + 1, self.hidden_size, num_layers=self.num_layers,
                               batch_first=False)
        elif self.rnn_type == 'LSTM':
            self.spatial_encoder = nn.LSTM(self.rnn_input_size, self.hidden_size, num_layers=self.num_layers,
                                batch_first=False)
        elif self.rnn_type == 'BiLSTM':
            self.spatial_encoder = nn.LSTM(self.rnn_input_size, self.hidden_size, num_layers=self.num_layers,
                                batch_first=False, bidirectional=True)
        else:
            raise ValueError("rnn_type should be ['GRU', 'LSTM', 'BiLSTM']")

        #spatial_adv
        # prototype layer
        self.prototypes = None
        if isinstance(self.k, list):
            self.prototypes = MultiPrototypes(self.hidden_size, self.k)
        elif self.k > 0:
            self.prototypes = nn.Linear(self.hidden_size, self.k, bias=False)

        # projection head
        self.projection_head = nn.Sequential(
            nn.Linear(self.hidden_size, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, self.hidden_size),
        )

        # dense layer
        self.s2st_projection = nn.Linear(self.hidden_size * self.bi, self.hidden_size * self.bi)
        if self.downstream == 'TUL':
            self.dense = nn.Linear(in_features=self.hidden_size * 2 * self.bi, out_features=self.user_size)
            self.t2st_projection = nn.Linear(self.hidden_size * self.bi, self.hidden_size * self.bi)
            self.projection = nn.Sequential(nn.Linear(self.hidden_size * self.bi, self.hidden_size * self.bi), nn.ReLU())
        elif self.downstream == 'POI':
            self.projection = nn.Sequential(
                nn.Linear(self.hidden_size * self.bi + self.user_emb_size, self.hidden_size * self.bi + self.user_emb_size),
                nn.ReLU())
            self.dense = nn.Linear(in_features=self.hidden_size * 2 * self.bi + self.user_emb_size, out_features=self.loc_size)
            self.t2st_projection = nn.Linear(self.hidden_size * self.bi + self.user_emb_size, self.hidden_size * self.bi)
        else:
            raise ValueError('downstream should in [TUL, POI]!')

        self.apply(self._init_weight)

    def _init_weight(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0)

    def spatial_encode(self, packed_stuff, all_len, cur_len, batch_size, momentum=False, downstream='POI'):
        if momentum == True:
            f_encoder = self.spatial_encoder_momentum
        else:
            f_encoder = self.spatial_encoder
        if self.rnn_type == 'GRU':
            spatial_out, h_n = f_encoder(packed_stuff)  # max_len*batch*hidden_size
        elif self.rnn_type == 'LSTM':
            spatial_out, (h_n, c_n) = f_encoder(packed_stuff)  # max_len*batch*hidden_size
        elif self.rnn_type == 'BiLSTM':
            spatial_out, (h_n, c_n) = f_encoder(packed_stuff)  # max_len*batch*hidden_size
        else :
            raise ValueError('rnn type is not in GRU, LSTM, BiLSTM! ')

        # unpack
        spatial_out, out_len = pad_packed_sequence(spatial_out, batch_first=False)
        spatial_out = spatial_out.permute(1, 0, 2)

        # out_len = all_len batch*max_len*hidden_size
        # concatenate
        if downstream == 'POI':
            final_out = spatial_out[0, (all_len[0] - cur_len[0]): all_len[0], :]
            for i in range(1, batch_size):
                final_out = torch.cat([final_out, spatial_out[i, (all_len[i] - cur_len[i]): all_len[i], :]], dim=0)
            # No longer concate user embedding
        elif downstream == 'TUL':
            final_out = spatial_out[0, (all_len[0] - 1): all_len[0], :]
            for i in range(1, batch_size):
                final_out = torch.cat([final_out, spatial_out[i, (all_len[i] - 1): all_len[i], :]], dim=0)
        else:
            raise ValueError('downstream is not in [POI, TUL]')
        return final_out

    def temporal_encode(self, packed_stuff, all_len, cur_len, batch_size, momentum=False, downstream='POI'):
        if momentum == True:
            f_encoder = self.temporal_encoder_momentum
        else:
            f_encoder = self.temporal_encoder
        if self.rnn_type == 'GRU':
            temporal_out, h_n = f_encoder(packed_stuff)  # max_len*batch*hidden_size
        elif self.rnn_type == 'LSTM':
            temporal_out, (h_n, c_n) = f_encoder(packed_stuff)  # max_len*batch*hidden_size
        elif self.rnn_type == 'BiLSTM':
            temporal_out, (h_n, c_n) = f_encoder(packed_stuff)  # max_len*batch*hidden_size
        else :
            raise ValueError('rnn type is not in GRU, LSTM, BiLSTM! ')

        # unpack
        temporal_out, out_len = pad_packed_sequence(temporal_out, batch_first=False)
        temporal_out = temporal_out.permute(1, 0, 2)

        # out_len = all_len batch*max_len*hidden_size
        # concatenate
        if downstream == 'POI':
            final_out = temporal_out[0, (all_len[0] - cur_len[0]): all_len[0], :]
            for i in range(1, batch_size):
                final_out = torch.cat([final_out, temporal_out[i, (all_len[i] - cur_len[i]): all_len[i], :]], dim=0)
            # No longer concate user embedding
        elif downstream == 'TUL':
            final_out = temporal_out[0, (all_len[0] - 1): all_len[0], :]
            for i in range(1, batch_size):
                final_out = torch.cat([final_out, temporal_out[i, (all_len[i] - 1): all_len[i], :]], dim=0)
        else:
            raise ValueError('downstream is not in [POI, TUL]')

        return final_out

    def get_params(self, decoder_input):
        """
        Generate model parameters based on the inputs
        Args:
            input: decoder input [batch, decoder_input_size]

        Returns:
            prior_logits: shape [batch, n_components]
            means: shape [batch, n_components]
            log_scales: shape [batch, n_components]
        """
        prior_logits, means, log_scales = self.hypernet(decoder_input)

        # Clamp values that go through exp for numerical stability
        prior_logits = clamp_preserve_gradients(prior_logits, self.min_clip, self.max_clip)
        log_scales = clamp_preserve_gradients(log_scales, self.min_clip, self.max_clip)

        # normalize prior_logits
        prior_logits = F.log_softmax(prior_logits, dim=-1)
        return prior_logits, means, log_scales

    def spatial_adv(self,inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        return self.spatial_adv_forward_head(inputs)

    def spatial_adv_forward_head(self, x):
        x = torch.cat((x[0],x[1]),dim=0)
        if self.projection_head is not None:
            x = self.projection_head(x)

        if self.l2norm:
            x = nn.functional.normalize(x, dim=1, p=2)

        if self.prototypes is not None:
            return x, self.prototypes(x)
        return x

    def distributed_sinkhorn(self, Q, nmb_iters):
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            Q /= sum_Q

            u = torch.zeros(Q.shape[0]).cuda(non_blocking=True)
            r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
            c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / (self.world_size * Q.shape[1])

            curr_sum = torch.sum(Q, dim=1)

            for it in range(nmb_iters):
                u = curr_sum
                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
                curr_sum = torch.sum(Q, dim=1)
            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    def forward(self, batch, mode='test', cont_conf=None, downstream='POI', queue = None, use_the_queue = False):

        self.spatial_encoder.flatten_parameters()
        self.temporal_encoder.flatten_parameters()
        self.temporal_encoder_momentum.flatten_parameters()
        
        if cont_conf is None:
            cont_conf = [0, 0, 0, 0]

        loc_cont, time_cont, user_cont, spatial_temporal_cont = cont_conf[0], cont_conf[1], cont_conf[2], cont_conf[3]
        loc = batch.X_all_loc
        # category vectors batch * seqlen * 768
        category_vectors = []
        for loc_seq in loc:
            category_vectors.append([self.category_vector[i] for i in loc_seq])
        category_vectors = torch.tensor(np.array(category_vectors)).to(self.device)
        tim = batch.X_all_tim
        user = batch.X_users
        geohash_ = batch.X_all_geohash
        cur_len = batch.target_lengths
        all_len = batch.X_lengths

        batch_size = loc.shape[0]
        loc_emb = self.emb_loc(loc)
        tim_emb = self.emb_tim(tim)
        user_emb = self.emb_user(user)
        geohash_ = self.geohash_dense(geohash_)
        category_vectors = self.category_dense(category_vectors)

        # concatenate and permute (make 'batch_first'=False)
        x = torch.cat([loc_emb, geohash_], dim=2)
        x = torch.cat([x, category_vectors.squeeze()], dim=2).permute(1, 0, 2)
        x_temporal = tim_emb.permute(1, 0, 2)

        # cat locs & taus
        all_tau = [torch.cat((batch.X_tau[0, :all_len[0] - cur_len[0]], batch.Y_tau[0, :cur_len[0]]), dim=-1)]
        self.truth_Y_tau = all_tau[0][all_len[0] - cur_len[0]:all_len[0]]

        for i in range(1, batch_size):
            # taus
            cur_tau = torch.cat((batch.X_tau[i, :all_len[i] - cur_len[i]], batch.Y_tau[i, :cur_len[i]]), dim=-1)
            all_tau.append(cur_tau)

            self.truth_Y_tau = torch.cat((self.truth_Y_tau, all_tau[i][all_len[i] - cur_len[i]:all_len[i]]), dim=0)

        all_tau = pad_sequence(all_tau, batch_first=False).to(self.device)
        x_temporal = torch.cat((all_tau.unsqueeze(-1), x_temporal), dim=-1)

        # pack
        pack_x = pack_padded_sequence(x, lengths=all_len, enforce_sorted=False)
        pack_x_temporal = pack_padded_sequence(x_temporal, lengths=all_len, enforce_sorted=False)

        final_spatial_out = self.spatial_encode(pack_x, all_len, cur_len, batch_size, downstream=downstream)
        final_temporal_out = self.temporal_encode(pack_x_temporal, all_len, cur_len, batch_size, downstream=downstream)

        all_user_emb = user_emb[0].unsqueeze(dim=0).repeat(cur_len[0], 1)
        for i in range(1, batch_size):
            all_user_emb = torch.cat([all_user_emb, user_emb[i].unsqueeze(dim=0).repeat(cur_len[i], 1)], dim=0)
        # concatenate
        if downstream == 'POI':
            prediction_out = torch.cat([final_spatial_out, final_temporal_out, all_user_emb], 1)
            dense = self.dense(prediction_out)  # Batch * loc_size
            pred = nn.LogSoftmax(dim=1)(dense)  # result
        elif downstream == 'TUL':
            prediction_out = torch.cat([final_spatial_out, final_temporal_out], 1)
            dense = self.dense(prediction_out)
            pred = nn.LogSoftmax(dim=1)(dense)  # result
        else:
            raise ValueError('downstream is not in [POI, TUL]')

        ####################   cont  start    #####################
        cont_loss_ls = [0, 0, 0, 0]
        if mode == 'train':
            # location contrast
            if loc_cont == 1:
                inputs = [final_spatial_out,self.dropout(final_spatial_out)]
                embedding, output = self.spatial_adv(inputs)
                embedding = embedding.detach()
                bs = inputs[0].size(0)
                loc_loss = 0
                for i, crop_id in enumerate(self.crops_for_assign):
                    with torch.no_grad():
                        out = output[bs * crop_id: bs * (crop_id + 1)]
                        # time to use the queue
                        if queue is not None:
                            if use_the_queue or not torch.all(queue[i, -1, :] == 0):
                                use_the_queue = True
                                out = torch.cat((torch.mm(
                                    queue[i],
                                    self.prototypes.weight.t()
                                ), out))
                            # fill the queue
                            queue[i, bs:] = queue[i, :-bs].clone()
                            queue[i, :bs] = embedding[crop_id * bs: (crop_id + 1) * bs]
                        # get assignments
                        q = torch.exp(out /self.epsilon).t()
                        with torch.no_grad():
                            Q=q
                            sum_Q = torch.sum(Q)
                            Q /= sum_Q
                            u = torch.zeros(Q.shape[0]).cuda(non_blocking=True)
                            r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
                            c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / (self.world_size * Q.shape[1])

                            curr_sum = torch.sum(Q, dim=1)
                            for it in range(3):
                                u = curr_sum
                                Q *= (r / u).unsqueeze(1)
                                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
                                curr_sum = torch.sum(Q, dim=1)
                            q = (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()[-bs:]
                    # cluster assignment prediction
                    subloss = 0
                    for v in np.delete(np.arange(np.sum(self.nmb_crops)), crop_id):
                        p = self.softmax(output[bs * v: bs * (v + 1)] / self.temperature)
                        subloss -= torch.mean(torch.sum(q * torch.log(p), dim=1))
                    loc_loss += subloss / (np.sum(self.nmb_crops) - 1)
                loc_loss /= len(self.crops_for_assign)

                cont_loss_ls[0] = loc_loss

            # time contrast
            if time_cont == 1:
                tim_emb_momentum = tim_emb
                x_temporal_momentum = tim_emb_momentum.permute(1, 0, 2)
                x_temporal_momentum = torch.cat((all_tau.unsqueeze(-1), x_temporal_momentum), dim=-1)

                pack_x_temporal_momentum = pack_padded_sequence(x_temporal_momentum, lengths=all_len, enforce_sorted=False)
                final_temporal_out_momentum = self.temporal_encode(pack_x_temporal_momentum, all_len, cur_len, batch_size, momentum=True, downstream=downstream)
                cos = nn.CosineSimilarity(dim=-1)
                cos_sim = cos(final_temporal_out_momentum.unsqueeze(1), final_temporal_out.unsqueeze(0))
                eye_mask = torch.eye(cos_sim.shape[0], device=cos_sim.device)
                reverse_eye_mask = torch.ones(cos_sim.shape[0], device=cos_sim.device) - eye_mask

                angle_cos = torch.acos(cos_sim)
                cos_sim1 = torch.cos(angle_cos + self.theta)
                cos_sim2 = (cos_sim * reverse_eye_mask + cos_sim1 * eye_mask)
                logits = cos_sim2 / self.temperature
                cont_crit_tim = nn.CrossEntropyLoss()
                labels = torch.arange(final_temporal_out.size(0), device=final_temporal_out.device)
                time_cont_loss = cont_crit_tim(logits, labels)
                cont_loss_ls[1] = time_cont_loss
            if user_cont == 1:
                pass
            if spatial_temporal_cont == 1:
                spatial_out = self.s2st_projection(final_spatial_out)
                if downstream != 'TUL':
                    temporal_out = self.t2st_projection(torch.cat([all_user_emb, final_temporal_out], 1))
                else:
                    temporal_out = self.t2st_projection(final_temporal_out)
                cos = nn.CosineSimilarity(dim=-1)

                cos_sim = cos(spatial_out.unsqueeze(1), temporal_out.unsqueeze(0))
                logits = cos_sim / self.temperature
                cont_crit_st = nn.CrossEntropyLoss()
                labels = torch.arange(spatial_out.size(0), device=spatial_out.device)
                st_cont_loss = cont_crit_st(logits, labels)
                cont_loss_ls[3] = st_cont_loss
        ####################   cont  end     #####################
        criterion = nn.NLLLoss().to(self.device)
        criterion1 = nn.L1Loss().to(self.device)


        if downstream == 'POI':
            s_loss_score = criterion(pred, batch.Y_location).requires_grad_(True)
            _, top_k_pred = torch.topk(pred, k=self.loc_size)  # (batch, K)=(batch, num_class)
        elif downstream == 'TUL':
            s_loss_score = criterion(pred, batch.X_users).requires_grad_(True)
            _, top_k_pred = torch.topk(pred, k=self.user_size)  # (batch, K)=(batch, num_class)
        else:
            raise ValueError('downstream is not in [POI, TUL]')

        if mode == 'train' and sum(cont_conf) != 0:
            return s_loss_score, top_k_pred, cont_loss_ls, queue
        else:
            return s_loss_score, top_k_pred, queue

class MultiPrototypes(nn.Module):
    def __init__(self, output_dim, nmb_prototypes):
        super(MultiPrototypes, self).__init__()
        self.nmb_heads = len(nmb_prototypes)
        for i, k in enumerate(nmb_prototypes):
            self.add_module("prototypes" + str(i), nn.Linear(output_dim, k, bias=False))

    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, "prototypes" + str(i))(x))
        return out