import torch
from torch.nn.utils.rnn import *

from model.utils import *
import numpy as np
class STCCR_TP_ModelConfig(DotDict):
    '''
    configuration of the STCCR_TP
    '''
    def __init__(self, loc_num=None, loc_emb_size=None,uid_size=None,user_emb_size=None, time_interval_minutes=None, tim_emb_size=None, uid_num=None,
                 uid_emb_size=None, hidden_size=None, rnn_type=None, num_layers=1, dropout=.0,  n_components=4, shift_init=0.0, scale_init=0.0, min_clip=-5.,
                 max_clip=3., hypernet_hidden_sizes=[],category_size=2, geohash_size=2,k=8,crops_for_assign=[0,1],nmb_crops=3,world_size=None,temperature=None,
                 epsilon=None,theta=None):
        super().__init__(),
        self.loc_num = loc_num 
        self.loc_emb_size = loc_emb_size
        self.uid_size = uid_size
        self.user_emb_size = user_emb_size
        self.tim_num = 7 * 24 * 60 // time_interval_minutes 
        self.tim_emb_size = tim_emb_size
        self.uid_num = uid_num
        self.uid_emb_size = uid_emb_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_components = n_components
        self.min_clip = min_clip
        self.max_clip = max_clip
        self.shift_init = shift_init
        self.scale_init = scale_init
        self.hypernet_hidden_sizes = hypernet_hidden_sizes
        self.decoder_input_size = uid_emb_size + hidden_size
        self.category_size = category_size
        self.geohash_size = geohash_size
        self.k = k
        self.crops_for_assign = crops_for_assign
        self.nmb_crops = nmb_crops 
        self.world_size = world_size 
        self.temperature = temperature 
        self.epsilon = epsilon 
        self.theta = theta



class STCCR_TP(nn.Module):

    def __init__(self, config):
        super(STCCR_TP, self).__init__()
        self.truth_Y_tau = None
        self.n_components = config.n_components
        self.min_clip = config.min_clip
        self.max_clip = config.max_clip
        self.shift_init = config.shift_init
        self.scale_init = config.scale_init

        self.loc_num = config.loc_num
        self.loc_emb_size = config.loc_emb_size
        self.tim_num = config.tim_num
        self.tim_emb_size = config.tim_emb_size
        self.uid_num = config.uid_num
        self.uid_emb_size = config.uid_emb_size
        self.hidden_size = config.hidden_size
        self.rnn_type = config.rnn_type
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        self.category_size = config['category_size']
        self.geohash_size = config['geohash_size']
        self.k = config['k']
        self.crops_for_assign = config['crops_for_assign']
        self.uid_size = config['uid_size']
        self.user_emb_size = config['user_emb_size']

        self.emb_loc = nn.Embedding(self.loc_num, self.loc_emb_size)
        self.emb_tim = nn.Embedding(self.tim_num, self.tim_emb_size)
        self.emb_uid = nn.Embedding(self.uid_num, self.uid_emb_size)

        self.linear1 = nn.Linear(self.hidden_size + self.uid_emb_size, (self.hidden_size + self.uid_emb_size)//4)
        self.linear2 = nn.Linear((self.hidden_size + self.uid_emb_size)//4, (self.hidden_size + self.uid_emb_size)//16)
        self.linear3 = nn.Linear((self.hidden_size + self.uid_emb_size)//16, 1)
        self.linear0 = nn.Linear((self.hidden_size)*2, self.hidden_size)

        self.Dropout = torch.nn.Dropout(config.dropout)
        self.sigmoid = nn.Sigmoid()
        self.mae = torch.nn.L1Loss()
        self.l2norm = True
        self.rnn_input_size = self.loc_emb_size + self.geohash_size + self.category_size
        input_size = self.loc_emb_size
        self.sinkhorn_iterations = 3
        self.crops_for_assign = config.crops_for_assign
        self.nmb_crops = config.nmb_crops
        self.world_size = config.world_size
        self.temperature = config.temperature
        self.epsilon = config.epsilon
        self.softmax = nn.Softmax()
        self.downstream = 'TP'
        self.theta = config.theta

        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, self.hidden_size, num_layers=self.num_layers, batch_first=True,
                              dropout=self.dropout)
            self.rnn_tau = nn.GRU(1, self.hidden_size, num_layers=self.num_layers, batch_first=True,
                                   dropout=self.dropout)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, self.hidden_size, num_layers=self.num_layers, batch_first=True,
                               dropout=self.dropout)
            self.rnn_tau = nn.LSTM(1, self.hidden_size, num_layers=self.num_layers, batch_first=True,
                               dropout=self.dropout)
        elif self.rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size, self.hidden_size, num_layers=self.num_layers, batch_first=True,
                              dropout=self.dropout)
        self.hypernet = Hypernet(config,
                                 hidden_sizes=config.hypernet_hidden_sizes,
                                 param_sizes=[config.n_components, config.n_components, config.n_components])

        # rnn layer
        if self.rnn_type == 'GRU':
            self.spatial_encoder = nn.GRU(self.rnn_input_size, self.hidden_size, num_layers=self.num_layers,
                                          batch_first=True)
            self.temporal_encoder = nn.GRU(1, self.hidden_size, num_layers=self.num_layers,
                                           batch_first=True)
            self.temporal_encoder_momentum = nn.GRU(1, self.hidden_size, num_layers=self.num_layers,
                                                    batch_first=True)
        elif self.rnn_type == 'LSTM':
            self.spatial_encoder = nn.LSTM(self.rnn_input_size, self.hidden_size, num_layers=self.num_layers,
                                           batch_first=True)
        elif self.rnn_type == 'BiLSTM':
            self.spatial_encoder = nn.LSTM(self.rnn_input_size, self.hidden_size, num_layers=self.num_layers,
                                           batch_first=True, bidirectional=True)
        else:
            raise ValueError("rnn_type should be ['GRU', 'LSTM', 'BiLSTM']")

        if self.rnn_type == 'BiLSTM':
            self.bi = 2
        else:
            self.bi = 1

        # dense layer
        self.s2st_projection = nn.Linear(self.hidden_size * self.bi, self.hidden_size * self.bi)
        if self.downstream == 'TP':
            self.projection = nn.Sequential(
                nn.Linear(self.hidden_size * self.bi + self.user_emb_size,
                          self.hidden_size * self.bi + self.user_emb_size),
                nn.ReLU())
            self.t2st_projection = nn.Linear(self.hidden_size * self.bi + self.user_emb_size,
                                             self.hidden_size * self.bi)
            final_in_size = self.hidden_size * self.bi * 2 + self.user_emb_size
            self.dense_s = nn.Linear(in_features=self.hidden_size,
                                     out_features=self.hidden_size)
            self.dense_t = nn.Linear(in_features=self.hidden_size * self.bi + self.user_emb_size,
                                     out_features=self.hidden_size * self.bi + self.user_emb_size)
            self.dense_st = nn.Linear(in_features=self.hidden_size * 2 * self.bi + self.user_emb_size,
                                      out_features=self.hidden_size * 2 * self.bi + self.user_emb_size)
            self.dense = nn.Sequential(
                nn.Linear(in_features=final_in_size, out_features=final_in_size // 4),
                nn.LeakyReLU(),
                nn.Linear(in_features=final_in_size // 4, out_features=final_in_size // 16),
                nn.LeakyReLU(),
                nn.Linear(in_features=final_in_size // 16, out_features=1),
            )

        self.apply(self._init_weight)

        # prototype layer
        self.prototypes = None
        if isinstance(self.k, list):
            self.prototypes = MultiPrototypes(self.hidden_size, self.k)
        elif self.k > 0:
            self.prototypes = nn.Linear(self.hidden_size, self.k, bias=False)

        self.projection_head1 = nn.Linear(self.hidden_size, 2048)
        self.projection_head2 = nn.BatchNorm1d(2048)
        self.projection_head3 = nn.ReLU(inplace=True)
        self.projection_head4 = nn.Linear(2048, self.hidden_size)

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

    def get_true_length(self, tau_i, loc_i, history_len, i):
        if len(torch.nonzero(tau_i)) > 0 and len(torch.nonzero(loc_i)) > 0:
            l_tau = torch.nonzero(tau_i)[-1] + 1
            l_loc = torch.nonzero(loc_i)[-1] + 1
            l = max(l_tau, l_loc)
        elif len(torch.nonzero(tau_i)) > 0:
            l = torch.nonzero(tau_i)[-1] + 1
        elif len(torch.nonzero(loc_i)) > 0:
            l = torch.nonzero(loc_i)[-1] + 1
        else:
            l = history_len[i]
        return l

    def temporal_encode(self, packed_stuff, all_len, cur_len, batch_size, momentum=False, downstream='TP'):
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

        # concatenate
        if downstream == 'TP':
            first = 0
            while   all_len[first]-cur_len[first]-2<0:
                first += 1
            final_temporal_out = temporal_out[first,  all_len[first]-cur_len[first]-2 : all_len[first]-2, :]

            for i in range(first + 1, batch_size):
                if all_len[i] - cur_len[i] - 2 >=0:
                    final_temporal_out = torch.cat([final_temporal_out, temporal_out[i, all_len[i]-cur_len[i]-2 : all_len[i]-2, :]], dim=0)
        return final_temporal_out
    
    def spatial_adv(self,inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        return self.spatial_adv_forward_head(inputs)

    def spatial_adv_forward_head(self, x):
        x = torch.cat((x[0],x[1]),dim=0)
        if self.projection_head1 is not None:
            x = torch.reshape(x,(-1,self.hidden_size))
            x = self.projection_head1(x)
            x = self.projection_head2(x)
            x = self.projection_head3(x)
            x = self.projection_head4(x)

        if self.l2norm:
            x = nn.functional.normalize(x, dim=1, p=2)

        if self.prototypes is not None:
            return x, self.prototypes(x)
        return x

    def forward(self, input, cont_conf, downstream='TP', mode = 'train', queue = None,use_the_queue = False):
        self.temporal_encoder.flatten_parameters()
        self.temporal_encoder_momentum.flatten_parameters()


        history_len = (input.X_length - 1).detach().cpu().long()
        batch_size = input.X_location.shape[0]
        cur_len = []
        for i in range(batch_size):
            cur_len.append(len(torch.nonzero(input.Y_tau[i])))
        cur_len = torch.tensor(cur_len).to(history_len.device)
        all_tau = [torch.cat((input.X_tau[0, :history_len[0]-cur_len[0]], input.Y_tau[0]), dim=-1)]
        all_loc = [torch.cat((input.X_location[0, :history_len[0]-cur_len[0]+1], input.Y_location[0]), dim=-1)]
        all_len = [self.get_true_length(all_tau[0], all_loc[0], history_len-cur_len, 0)]

        for i in range(1, batch_size):
            # taus
            cur_tau = torch.cat((input.X_tau[i, :history_len[i]-cur_len[i]], input.Y_tau[i]), dim=-1)
            all_tau.append(cur_tau)

            #locs
            cur_loc = torch.cat((input.X_location[i, :history_len[i]-cur_len[i]+1],  input.Y_location[i]), dim=-1)
            all_loc.append(cur_loc)
            all_len.append(self.get_true_length(all_tau[i], all_loc[i], history_len-cur_len, i))

        all_tau = pad_sequence(all_tau, batch_first=True)
        all_loc = pad_sequence(all_loc, batch_first=True)
        all_len = torch.LongTensor(all_len)
        all_loc_emb = self.emb_loc(all_loc)
        x = all_loc_emb
        packed_x = pack_padded_sequence(x, lengths=all_len, batch_first=True, enforce_sorted=False)
        packed_x_tau =pack_padded_sequence(all_tau.unsqueeze(-1), lengths=all_len-1, batch_first=True, enforce_sorted=False)
        if self.rnn_type == 'GRU':
            output, (h_n, _) = self.rnn(packed_x)  # h_n: (num_layers * num_directions, batch, hidden_size)
            output_tau, (h_n, _) = self.rnn_tau(packed_x_tau)
        else:
            output, h_n = self.rnn(packed_x)

        uid_emb = self.emb_uid(input.X_user)

        # unpack
        output, out_len = pad_packed_sequence(output, batch_first=True)
        output_tau, out_len = pad_packed_sequence(output_tau, batch_first=False)
        spatial_out = output
        temporal_out = output_tau.permute(1, 0, 2)
        # concatenate
        first = 0
        while all_len[first]-cur_len[first]-2<0:
            first += 1
        final_spatial_out = spatial_out[first, all_len[first]-cur_len[first]-1 : all_len[first]-1, :]
        final_temporal_out = temporal_out[first,all_len[first]-cur_len[first]-2 : all_len[first]-2, :]
        all_user_emb = uid_emb[first].unsqueeze(dim=0).repeat(cur_len[first], 1)
        self.truth_Y_tau = input.Y_tau[first, :cur_len[first]]
        for i in range(first + 1, batch_size):
            if all_len[i]-cur_len[i]-2>=0:
                final_spatial_out = torch.cat([final_spatial_out, spatial_out[i, all_len[i]-cur_len[i]-1 : all_len[i]-1, :]], dim=0)
                final_temporal_out = torch.cat([final_temporal_out, temporal_out[i, all_len[i]-cur_len[i]-2 : all_len[i]-2, :]], dim=0)
                all_user_emb = torch.cat([all_user_emb, uid_emb[i].unsqueeze(dim=0).repeat(cur_len[i], 1)],dim=0)
                self.truth_Y_tau = torch.cat((self.truth_Y_tau, input.Y_tau[i, :cur_len[i]]), dim=0)
            else: pass


        final_out = torch.cat((final_spatial_out, final_temporal_out), dim=-1)
        final_out = self.linear0(final_out)
        final_out = torch.cat([final_out, all_user_emb], dim=-1)
        new_batch_size = final_out.shape[0]
        loc_cont, time_cont, user_cont, spatial_temporal_cont = cont_conf[0], cont_conf[1], cont_conf[2], cont_conf[3]
        ####################   cont  start    #####################
        cont_loss_ls = [0, 0, 0, 0]
        if mode == 'train':
            # location contrast
            if loc_cont == 1:
                inputs = [final_spatial_out, self.Dropout(final_spatial_out)]
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
                        q = torch.exp(out / self.epsilon).t()

                        with torch.no_grad():
                            Q = q
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
                    subloss = 0
                    for v in np.delete(np.arange(np.sum(self.nmb_crops)), crop_id):
                        p = self.softmax(output[bs * v: bs * (v + 1)] / self.temperature)
                        subloss -= torch.mean(torch.sum(q * torch.log(p), dim=1))
                    loc_loss += subloss / (np.sum(self.nmb_crops) - 1)
                loc_loss /= len(self.crops_for_assign)
                cont_loss_ls[0] = loc_loss

            # time contrast
            if time_cont == 1:
                x_temporal_momentum = all_tau.unsqueeze(-1)
                final_temporal_out_momentum = self.temporal_encode(packed_x_tau, all_len, cur_len,
                                                                   batch_size, momentum=True, downstream=downstream)
                cos = nn.CosineSimilarity(dim=-1)
                cos_sim = cos(final_temporal_out_momentum.unsqueeze(1), final_temporal_out.unsqueeze(0))  # bs * bs
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
                temporal_out = self.t2st_projection(torch.cat([all_user_emb, final_temporal_out], 1))
                cos = nn.CosineSimilarity(dim=-1)
                cos_sim = cos(spatial_out.unsqueeze(1), temporal_out.unsqueeze(0))
                logits = cos_sim / self.temperature
                cont_crit_st = nn.CrossEntropyLoss()
                labels = torch.arange(spatial_out.size(0), device=spatial_out.device)
                st_cont_loss = cont_crit_st(logits, labels)
                cont_loss_ls[3] = st_cont_loss
        ####################   cont  end     #####################

        y = torch.log(self.truth_Y_tau + 1e-2).unsqueeze(-1)
        y = (y - self.shift_init.to(y.device)) / self.scale_init.to(y.device)
        mean_time = self.linear3(self.sigmoid(self.linear2(self.sigmoid(self.linear1(final_out)))))
        loss = self.mae(mean_time, y)
        a = self.scale_init.to(y.device)
        b = self.shift_init.to(y.device)
        mean_time = torch.exp(a * mean_time + b )
        if mode == 'train' and sum(cont_conf) != 0:
            return loss, mean_time, cont_loss_ls, queue, self.truth_Y_tau.detach().cpu().numpy()
        else:
            return loss, mean_time, queue, self.truth_Y_tau.detach().cpu().numpy()



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