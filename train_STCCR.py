import argparse
import configparser
import preprocess.load_data_for_STCCR as preprocess 
from model.STCCR import *
from copy import deepcopy
from utils import *
from torch.nn.utils import clip_grad_norm_
import torch
import torch.nn as nn
import time


# read hyper-param settings
parser = argparse.ArgumentParser()
parser.add_argument("--config", default='config/STCCR_gow_nyc_TUL.conf', type=str,
                    help="configuration file path")
parser.add_argument("--dataroot", default='./STCCR_data/', type=str,
                    help="data root directory")
args = parser.parse_args()
config_file = args.config
data_root = args.dataroot
config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config))
print('>>>>>>>  configuration   <<<<<<<')
with open(config_file, 'r') as f:
    print(f.read())
print('\n')
config.read(args.config)
data_config = config['Data']
training_config = config['Training']
model_config = config['Model']

# Data config
dataset_name = data_config['dataset_name']
max_his_period_days = data_config['max_his_period_days']
max_merge_seconds_limit = data_config['max_merge_seconds_limit']
max_delta_mins = data_config['max_delta_mins']
min_session_mins = data_config['min_session_mins']
least_disuser_count = data_config['least_disuser_count']
least_checkins_count = data_config['least_checkins_count']
latN = data_config['latN']
lngN = data_config['lngN']
split_save = bool(int(data_config['split_save']))
dataset_name = dataset_name + '_' + max_his_period_days + 'H' + max_merge_seconds_limit + 'M' + max_delta_mins + 'd' + min_session_mins + 's' + least_disuser_count + 'P' + least_checkins_count + 'U'

# Training config
mode = training_config['mode'].strip()
ctx = training_config['ctx']
os.environ["CUDA_VISIBLE_DEVICES"] = ctx
USE_CUDA = torch.cuda.is_available()
print("CUDA:", USE_CUDA, ctx)
device = torch.device("cuda" if USE_CUDA else "cpu")
print('device:', device)
use_nni = bool(int(training_config['use_nni']))
regularization = float(training_config['regularization'])
learning_rate = float(training_config['learning_rate'])
max_epochs = int(training_config['max_epochs'])
display_step = int(training_config['display_step'])
patience = int(training_config['patience'])
train_batch = int(training_config['train_batch'])
val_batch = int(training_config['val_batch'])
test_batch = int(training_config['test_batch'])
batch_size = int(training_config['batch_size'])
save_results = bool(int(training_config['save_results']))

specific_config = 'STCCR'

# Model Setting
loc_emb_size = int(model_config['loc_emb_size'])
tim_emb_size = int(model_config['tim_emb_size'])
user_emb_size = int(model_config['user_emb_size'])
hidden_size = int(model_config['hidden_size'])

category_size = int(model_config['category_size'])
geohash_size = int(model_config['geohash_size'])

adv = int(model_config['adv'])
rnn_type = model_config['rnn_type']
num_layers = int(model_config['num_layers'])
downstream = model_config['downstream']

loc_noise_mean = float(model_config['loc_noise_mean'])
loc_noise_sigma = float(model_config['loc_noise_sigma'])
tim_noise_mean = float(model_config['tim_noise_mean'])
tim_noise_sigma = float(model_config['tim_noise_sigma'])
user_noise_mean = float(model_config['user_noise_mean'])
user_noise_sigma = float(model_config['user_noise_sigma'])
tau = float(model_config['tau'])
pos_eps = float(model_config['pos_eps'])
neg_eps = float(model_config['neg_eps'])
dropout_rate_1 = float(model_config['dropout_rate_1'])
dropout_rate_2 = dropout_rate_1

momentum = float(model_config['momentum'])
theta = float(model_config['theta'])
temperature = float(model_config['temperature'])
k = int(model_config['k'])
self_weight_s = float(model_config['self_weight_s'])
self_weight_t = float(model_config['self_weight_t'])
self_weight_st = float(model_config['self_weight_st'])

dump_path = 'checkpoints'
rank = model_config['rank']
epoch_queue_starts = int(model_config['epoch_queue_starts'])
crops_for_assign = [0,1]
feat_dim = int(model_config['feat_dim'])
queue_length = int(model_config['queue_length'])
world_size = int(model_config['world_size'])
epsilon = float(model_config['epsilon'])
dropout_spatial = float(model_config['dropout_spatial'])

if use_nni:
    import nni
    param = nni.get_next_parameter()
    # multi-dataset
    batch_size = int(param['batch_size'])
    hidden_size = int(param['hidden_size'])
    user_emb_size = int(param['user_emb_size'])
    category_size = int(param['category_size'])
    geohash_size = int(param['geohash_size'])
    num_layers = int(param['num_layers'])
    momentum = float(param['momentum'])
    theta = float(param['theta'])
    temperature = float(param['temperature'])
    k = int(param['k'])
    self_weight_s = float(param['self_weight_s'])
    self_weight_t = float(param['self_weight_t'])
    self_weight_st = float(param['self_weight_st'])
    epsilon = float(param['epsilon'])
    dropout_spatial = float(param['dropout_spatial'])

train_batch = batch_size
val_batch = batch_size
test_batch = batch_size

print('load dataset:', dataset_name)
print('split_save:', split_save)

# Data

print('Loading data & Category vector...')
data_train, data_val, data_test, feature_category, feature_lat, feature_lng, latN, lngN, category_cnt, category_vector = preprocess.load_dataset_for_STCCR(
    dataset_name, save_split=split_save, data_root=data_root, device=device)
print("feature_category: ", feature_category.shape,
      feature_category)  # feature_category[venue's index] -> venue's category's index

# Set the parameters for affine normalization layer depending on the decoder ()
trainY_tau_mean, trainY_tau_std = data_train.get_tau_log_mean_std_Y()
print('trainY_tau_mean:', trainY_tau_mean, flush=True)
print('trainY_tau_std:', trainY_tau_std, flush=True)

collate = preprocess.collate_session_based  # padding sequence with variable len

dl_train = torch.utils.data.DataLoader(data_train, batch_size=train_batch, shuffle=True,
                                       collate_fn=collate, drop_last=True)  
dl_val = torch.utils.data.DataLoader(data_val, batch_size=val_batch, shuffle=False, collate_fn=collate, drop_last=True)
dl_test = torch.utils.data.DataLoader(data_test, batch_size=test_batch, shuffle=False, collate_fn=collate, drop_last=True)


# Model setup
print('Building model...', flush=True)
# General model config
tim_size = 48  
fill_value = 0  
use_semantic = True  
if use_semantic:
    print("Use semantic information from venue name!")
else:
    print("Don't use semantic information from venue name!")
general_config = STCCR_ModelConfig(loc_size=int(data_train.venue_cnt), tim_size=tim_size,
                                  uid_size=int(data_train.user_cnt), tim_emb_size=tim_emb_size,
                                  loc_emb_size=loc_emb_size, hidden_size=hidden_size, user_emb_size=user_emb_size,
                                  device=device,
                                  geohash_size=geohash_size, category_size=category_size,
                                  loc_noise_mean=loc_noise_mean, loc_noise_sigma=loc_noise_sigma,
                                  tim_noise_mean=tim_noise_mean, tim_noise_sigma=tim_noise_sigma,
                                  user_noise_mean=user_noise_mean, user_noise_sigma=user_noise_sigma, tau=tau,
                                  momentum=momentum, k=k, theta=theta, temperature=temperature,
                                  pos_eps=pos_eps, neg_eps=neg_eps, dropout_rate_1=dropout_rate_1,
                                  dropout_rate_2=dropout_rate_2, category_vector=category_vector, rnn_type=rnn_type, num_layers=num_layers, downstream=downstream,
                                  max_delta_mins=max_delta_mins, dropout_spatial = dropout_spatial,
                                  epsilon=epsilon)
# Define model
model = STCCR(general_config).to(device)
print(model, flush=True)

params_path = os.path.join('experiments', dataset_name.replace('(', '').replace(')', ''), specific_config)
print('params_path:', params_path)

if use_nni:
    exp_id = nni.get_experiment_id()
    trail_id = nni.get_trial_id()
    best_name = str(exp_id) + '.' + str(trail_id) + downstream + 'best.params'
    params_filename = os.path.join(params_path, best_name)
else:
    best_name = 'best.params'
    params_filename = os.path.join(params_path, best_name)

if mode == 'train':
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    for p_t in model.temporal_encoder_momentum.parameters():
        p_t.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("total_params:", total_params, flush=True)
    print("total_trainable_params:", total_trainable_params, flush=True)

    if os.path.exists(params_path):
        print('already exist %s' % (params_path), flush=True)
    else:
        os.makedirs(params_path)
        print('create params directory %s' % (params_path), flush=True)

    print('Starting training...', flush=True)

    # build the queue
    queue = None
    queue_path = os.path.join(dump_path, "queue" + rank + ".pth")

    impatient = 0
    best_hit20 = -np.inf
    best_tnll = np.inf
    best_model = deepcopy(model.state_dict())
    global_step = 0
    best_epoch = -1
    opt = torch.optim.Adam(model.parameters(), weight_decay=regularization, lr=learning_rate, amsgrad=True)
    start = time.time()
    for epoch in range(0, max_epochs):
        model.train()
        batch_cnt = 0
        # optionally starts a queue
        if queue_length > 0 and epoch >= epoch_queue_starts and queue is None:
            queue = torch.zeros(
                len(crops_for_assign),
                -queue_length // world_size,
                feat_dim,
            ).cuda()
        for input in dl_train:
            opt.zero_grad()
            # cosine similarity matrix can be big (up to 5GiB), consider cutting.
            batch_cnt += 1
            if input.X_all_loc.shape[1] >= 700:
                print(f'batch: {batch_cnt}, length: {input.X_all_loc.shape[1]}')
                continue
            if adv == 1:
                s_loss_score, top_k_pred, cont_loss_ls, queue = model(input, mode='train', downstream=downstream, cont_conf=[1, 1, 1, 1], queue = queue)
                loss_total = (1 - self_weight_s - self_weight_t - self_weight_st) * s_loss_score + cont_loss_ls[0] * self_weight_s + cont_loss_ls[1] * self_weight_t + cont_loss_ls[3] * self_weight_st
            else:
                s_loss_score, top_k_pred, queue = model(input, mode='train', downstream=downstream, queue = queue)
                loss_total = s_loss_score

            loss_total.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            # momentum update
            if model.momentum > 0:
                for param_momentum, param in zip(model.temporal_encoder_momentum.parameters(), model.temporal_encoder.parameters()):
                    param_momentum.data = param_momentum.data * model.momentum + (1. - model.momentum) * param.data

            global_step += 1
            if downstream == 'POI':
                ys = input.Y_location
            elif downstream == 'TUL':
                ys = input.X_users  # (batch,)
            else:
                raise ValueError('downstream is not in [POI, TUL]')

            loss_total = loss_total.item()
            hit_ratio, mrr = evaluate_location(ys.cpu().numpy(), top_k_pred.cpu().numpy())  # [k]
            if queue is not None:
                torch.save({"queue": queue}, queue_path)

        model.eval()
        with torch.no_grad():
            all_loss_s_val, hit_ratio_val, mrr_val = get_s_baselines_total_loss_s_for_STCCR_DOWN(dl_val, model, downstream=downstream)
            if (hit_ratio_val[19] - best_hit20) < 1e-4:
                impatient += 1
                if best_hit20 < hit_ratio_val[19]:
                    best_hit20 = hit_ratio_val[19]
                    best_model = deepcopy(model.state_dict())
                    best_epoch = epoch
            else:
                best_hit20 = hit_ratio_val[19]
                best_model = deepcopy(model.state_dict())
                best_epoch = epoch
                impatient = 0

            if impatient >= patience:
                print('Breaking due to early stopping at epoch %d,best epoch at %d' % (epoch, best_epoch), flush=True)
                break

            if epoch % display_step == 0:
                if adv == 1:
                    print('Epoch %4d, train_loss=%.4f, spatial_loss=%.4f, temporal_loss=%.4f, val_loss=%.4f, val_mrr=%.4f, val_hit_1=%.4f, val_hit_20=%.4f' % (
                    epoch, loss_total, cont_loss_ls[0], cont_loss_ls[1], all_loss_s_val, mrr_val, hit_ratio_val[0], hit_ratio_val[19]), flush=True)
                else:
                    print('Epoch %4d, train_loss=%.4f, val_loss=%.4f, val_mrr=%.4f, val_hit_1=%.4f, val_hit_20=%.4f' % (
                    epoch, loss_total, all_loss_s_val, mrr_val, hit_ratio_val[0], hit_ratio_val[19]), flush=True)

            if use_nni:
                nni.report_intermediate_result(hit_ratio_val[19])

        torch.save(best_model, params_filename)

    print("best epoch at %d" % best_epoch, flush=True)
    print('save parameters to file: %s' % params_filename, flush=True)
    print("training time: ", time.time() - start)

### Evaluation
print('----- test ----')
model.load_state_dict(torch.load(params_filename))
model.eval()
with torch.no_grad():
    train_all_loss_s, train_hit_ratio, train_mrr = get_s_baselines_total_loss_s_for_STCCR_DOWN(dl_train, model, downstream=downstream)
    val_all_loss_s, val_hit_ratio, val_mrr = get_s_baselines_total_loss_s_for_STCCR_DOWN(dl_val, model, downstream=downstream)
    test_all_loss_s, test_hit_ratio, test_mrr = get_s_baselines_total_loss_s_for_STCCR_DOWN(dl_test, model, downstream=downstream)

    print('Dataset\t loss\t hit_1\t hit_3\t hit_5\t hit_7\t hit_10\t hit_15\t hit_20\t MRR\t\n' +
            'Train:\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t\n' % (
            train_all_loss_s, train_hit_ratio[0], train_hit_ratio[2], train_hit_ratio[4], train_hit_ratio[6],
            train_hit_ratio[9], train_hit_ratio[14], train_hit_ratio[19], train_mrr) +
            'Val:\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t\n' % (
            val_all_loss_s, val_hit_ratio[0], val_hit_ratio[2], val_hit_ratio[4], val_hit_ratio[6], val_hit_ratio[9],
            val_hit_ratio[14], val_hit_ratio[19], val_mrr) +
            'Test:\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t\n' % (
            test_all_loss_s, test_hit_ratio[0], test_hit_ratio[2], test_hit_ratio[4], test_hit_ratio[6],
            test_hit_ratio[9], test_hit_ratio[14], test_hit_ratio[19], test_mrr), flush=True)

    if use_nni:
        nni.report_final_result(val_hit_ratio[19])
