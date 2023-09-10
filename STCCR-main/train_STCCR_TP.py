import argparse
import configparser
import shutil
from copy import deepcopy
from torch.nn.utils import clip_grad_norm_
from model.STCCR_TP import *
import preprocess.load_data_for_STCCR_TP as preprocess
from utils import *
import warnings
# import nni
import torch

warnings.filterwarnings("ignore")
config_file = 'config/STCCR_gow_nyc_TP.conf'
# read hyper-param settings
parser = argparse.ArgumentParser()
parser.add_argument("--config", default=config_file, type=str, help="configuration file path")
parser.add_argument("--dataroot", default='./STCCR_data/', type=str,
                    help="data root directory")
args = parser.parse_args()
data_root = args.dataroot
config_file = args.config
config = configparser.ConfigParser()
print('Read configuration file: %s' % config_file)
print('>>>>>>>  configuration   <<<<<<<')
with open(config_file, 'r') as f:
    print(f.read())
print('\n')
config.read(args.config)
data_config = config['Data']
training_config = config['Training']
encoder_config = config['Encoder']
decoderT_config = config['DecoderT']

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
print('load dataset:', dataset_name)
print('split_save:', split_save)
experiment_base_dir = None

# Training config
use_nni = bool(int(training_config['use_nni']))
save_results = bool(int(training_config['save_results']))
adv = int(training_config['adv'])
self_weight_s = float(training_config['self_weight_s'])
self_weight_t = float(training_config['self_weight_t'])
self_weight_st = float(training_config['self_weight_st'])
dump_path = training_config['dump_path']
rank = training_config['rank']
epoch_queue_starts = int(training_config['epoch_queue_starts'])
crops_for_assign = [0,1]
feat_dim = int(training_config['feat_dim'])
queue_length = int(training_config['queue_length'])
world_size = int(training_config['world_size'])
nmb_crops = [2]
epsilon = float(training_config['epsilon'])
theta = float(training_config['theta'])
temperature = float(training_config['temperature'])

if use_nni:
    import nni
    param = nni.get_next_parameter()


mode = training_config['mode'].strip()
ctx = training_config['ctx']
os.environ["CUDA_VISIBLE_DEVICES"] = ctx
USE_CUDA = torch.cuda.is_available()
print('torch.cuda.device_count():', torch.cuda.device_count())
print("CUDA:", USE_CUDA, ctx)
device = torch.device("cuda" if USE_CUDA else "cpu")
localGPU = device
regularization = float(training_config['regularization'])
learning_rate = float(training_config['learning_rate'])
max_epochs = int(training_config['max_epochs'])
display_step = int(training_config['display_step'])
patience = int(training_config['patience'])
train_batch = int(training_config['train_batch'])
val_batch = int(training_config['val_batch'])
test_batch = int(training_config['test_batch'])
rnn_type = encoder_config['rnn_type']
specific_config = 'STCCR_TP'
time_interval_minutes = int(encoder_config['time_interval_minutes'])

loc_emb_size = int(encoder_config['loc_emb_size'])
tim_emb_size = int(encoder_config['tim_emb_size'])
uid_emb_size = int(encoder_config['uid_emb_size'])
hidden_size = int(encoder_config['hidden_size'])
num_of_rnn_layers = int(encoder_config['num_of_rnn_layers'])
# decoder T
n_components = int(decoderT_config['n_components'])
if use_nni:
    import nni
    batch_size = int(param['batch_size'])
    hidden_size = int(param['hidden_size'])
    user_emb_size = int(param['user_emb_size'])
    category_size = int(param['category_size'])
    geohash_size = int(param['geohash_size'])
    momentum = float(param['momentum'])
    theta = float(param['theta'])
    temperature = float(param['temperature'])
    k = int(param['k'])
    self_weight_s = float(param['self_weight_s'])
    self_weight_t = float(param['self_weight_t'])
    self_weight_st = float(param['self_weight_st'])
    downstream = param['downstream']
    epsilon = float(param['epsilon'])
    dropout_spatial = float(param['dropout_spatial'])
    learning_rate = float(param['learning_rate'])

# Model
dropout = float(encoder_config['dropout'])
decoder_input_size = uid_emb_size + hidden_size

# Data
print('Loading data...', flush=True)
data_train, data_val, data_test, gts, gss, time_info, week_info, feature_category, feature_lat, feature_lng, latN, lngN, category_cnt, feature_lat_ori, feature_lng_ori, distance, guassian_distance_matrix, user_lidfreq, min_len_x, max_len_x\
    = preprocess.load_data_for_STCCR_TP(dataset_name, data_root, split_save, time_interval_minutes=time_interval_minutes,  proximity_T=0, proximity_S=0, semantic_T=0, semantic_S=0, thetaSS=0, thetaTS_T=0, thetaTS_S=0, localGPU=localGPU)

print('feature_lat_ori[:10]:', feature_lat_ori[:10])
print('feature_lng_ori[:10]:', feature_lng_ori[:10])

# Set the parameters for affine normalization layer depending on the decoder (see Appendix D.3 in the paper)
trainY_tau_mean, trainY_tau_std = data_train.get_tau_log_mean_std_Y()
print('trainY_tau_mean:', trainY_tau_mean, flush=True)
print('trainY_tau_std:', trainY_tau_std, flush=True)

collate = preprocess.collate_session_based  # padding sequence with variable length

dl_train = torch.utils.data.DataLoader(data_train, batch_size=train_batch, shuffle=True, collate_fn=collate)  # 调用专门的collate_fn进行封装
dl_val = torch.utils.data.DataLoader(data_val, batch_size=val_batch, shuffle=False, collate_fn=collate)
dl_test = torch.utils.data.DataLoader(data_test, batch_size=test_batch, shuffle=False, collate_fn=collate)

# Model setup
print('Building model...', flush=True)
# General model config
general_config = STCCR_TP_ModelConfig(loc_num=int(data_train.venue_cnt), loc_emb_size=loc_emb_size,uid_size=int(data_train.user_cnt),user_emb_size=uid_emb_size,
                                    time_interval_minutes=time_interval_minutes, tim_emb_size=tim_emb_size, uid_num=int(data_train.user_cnt), uid_emb_size=uid_emb_size,
                                    hidden_size=hidden_size, rnn_type=rnn_type, num_layers=num_of_rnn_layers, dropout=dropout,  n_components=n_components,
                                    shift_init=trainY_tau_mean, scale_init=trainY_tau_std,crops_for_assign=crops_for_assign,nmb_crops=nmb_crops,world_size=world_size,temperature=temperature,
                                    epsilon=epsilon,theta=theta)

# Define model
model = STCCR_TP(general_config).cuda()
print(model, flush=True)

params_path = os.path.join('experiments', dataset_name.replace('(', '').replace(')', ''), specific_config)
print('params_path:', params_path, flush=True)

## in train mode
if mode == 'train':
    # parameter initialization
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("total_params:", total_params, flush=True)
    print("total_trainable_params:", total_trainable_params, flush=True)

    if os.path.exists(params_path):
        shutil.rmtree(params_path)
        os.makedirs(params_path)
        print('delete the old one and create params directory %s' % (params_path), flush=True)
    else:
        os.makedirs(params_path)
        print('create params directory %s' % (params_path), flush=True)

    print('Starting training...', flush=True)

    # build the queue
    queue = None
    queue_path = os.path.join(dump_path, "queue" + rank + ".pth")

    impatient = 0
    best_tnll = np.inf
    best_model = deepcopy(model.state_dict())
    global_step = 0
    best_epoch = -1
    params_filename = os.path.join(params_path, 'best.params')

    ## training
    # Define optimizer
    opt = torch.optim.Adam(model.parameters(), weight_decay=regularization, lr=learning_rate, amsgrad=True)

    for epoch in range(0, max_epochs):
        model.train()
        # optionally starts a queue
        if queue_length > 0 and epoch >= epoch_queue_starts and queue is None:
            queue = torch.zeros(
                len(crops_for_assign),
                -queue_length // world_size,
                feat_dim,
            ).cuda()
        for i, input in enumerate(dl_train):
            opt.zero_grad()
            if adv == 1:
                s_loss_score, top_k_pred, cont_loss_ls, queue,_ = model(input, mode='train', downstream='TP', cont_conf=[1, 1, 1, 1], queue = queue)
                loss_total = (1 - self_weight_s - self_weight_t - self_weight_st) * s_loss_score + cont_loss_ls[0] * self_weight_s + cont_loss_ls[1] * self_weight_t + cont_loss_ls[3] * self_weight_st
            else:
                s_loss_score, top_k_pred, queue,_ = model(input, mode='train', downstream='TP', cont_conf=[0, 0, 0, 0], queue = queue)
                loss_total = s_loss_score
            loss_total.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            global_step += 1

        model.eval()
        with torch.no_grad():
            mae_val, mape_val, rmse_val, nll_t_val = get_t_for_STCCR_TP_adv(dl_val, model)
            if (best_tnll - nll_t_val) < 1e-4:
                impatient += 1
                if nll_t_val < best_tnll:
                    best_tnll = nll_t_val
                    best_model = deepcopy(model.state_dict())
                    best_epoch = epoch
            else:
                best_tnll = nll_t_val
                best_model = deepcopy(model.state_dict())
                best_epoch = epoch
                impatient = 0

            if impatient >= patience:
                print('Breaking due to early stopping at epoch %d， best epoch at %d' % (epoch, best_epoch), flush=True)
                break

            if (epoch) % display_step == 0:
                    print('Epoch %4d, train_tnll=%.4f, val_tnll=%.4f, val_mae=%.4f, val_rmse=%.4f, val_nll=%.4f' % (epoch, s_loss_score, nll_t_val, mae_val,  rmse_val, mape_val), flush=True)
            if use_nni:
                    nni.report_intermediate_result(mae_val)

        torch.save(best_model, params_filename)

    print("best epoch at %d" % best_epoch, flush=True)
    print('save parameters to file: %s' % params_filename, flush=True)

### Evaluation
print('----- test ----')
params_filename = os.path.join(params_path, 'best.params')
print('load model from:', params_filename)
model.load_state_dict(torch.load(params_filename))
model.eval()
with torch.no_grad():
    print('evaluate on the train set ... ')
    train_mae, train_mape, train_rmse, train_nll_t = get_t_for_STCCR_TP_adv(dl_train, model, save_filename='train', params_path=params_path, experiment_base_dir=experiment_base_dir, use_nni=use_nni)
    print('evaluate on the val set ... ')
    val_mae, val_mape, val_rmse, val_nll_t = get_t_for_STCCR_TP_adv(dl_val, model, save_filename='val', params_path=params_path, experiment_base_dir=experiment_base_dir, use_nni=use_nni)
    print('evaluate on the test set ... ')
    test_mae, test_mape, test_rmse, test_nll_t = get_t_for_STCCR_TP_adv(dl_test, model, save_filename='test', params_path=params_path, experiment_base_dir=experiment_base_dir, use_nni=use_nni)


    print('Dataset\t MAE\t RMSE\t MAPE\t TNll\t\n' +
          'Train:\t %.4f\t %.4f\t %.4f\t %.4f\t\n' % (train_mae, train_rmse, train_mape, train_nll_t) +
          'Val:\t %.4f\t %.4f\t %.4f\t %.4f\t\n' % (val_mae, val_rmse, val_mape, val_nll_t) +
          'Test:\t %.4f\t %.4f\t %.4f\t %.4f\t\n' % (test_mae, test_rmse, test_mape, test_nll_t), flush=True)

if use_nni:
    nni.report_final_result(test_mae)



