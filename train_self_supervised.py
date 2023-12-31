import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path
import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from evaluation.evaluation import eval_edge_prediction, eval_recommendation
from model.tgn import TGN
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
from utils.data_processing import get_data, compute_time_statistics

torch.manual_seed(0)
np.random.seed(0)

### Argument and global variables
parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)', default='wikipedia')
parser.add_argument('--bs', type=int, default=200, help='Batch_size')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=1, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=1, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to backprop')
parser.add_argument('--use_memory', action='store_true',
                    help='Whether to augment the model with a node memory')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
  "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=[
  "mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", choices=[
  "gru", "rnn"], help='Type of memory updater')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message '
                                                                        'aggregator')
parser.add_argument('--memory_update_at_end', action='store_true',
                    help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for '
                                                                'each user')
parser.add_argument('--different_new_nodes', action='store_true',
                    help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true',
                    help='Whether to randomize node features')
parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the source node as part of the message')
parser.add_argument('--dyrep', action='store_true',
                    help='Whether to run the dyrep model')

try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
# NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}.pth'
get_checkpoint_path = lambda \
    epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}.pth'

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
Path("log/").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

### Extract data for training, validation and testing
node_features, edge_features, \
full_data, train_data, val_data, test_data, \
new_node_val_data, new_node_test_data = get_data(DATA,
                                                  different_new_nodes_between_val_and_test=args.different_new_nodes, 
                                                  randomize_features=args.randomize_features)

### get the final_time_feature.pkl dictionary # ts별로 모든 주식의 과거 일별 가격 모아 둔 딕셔너리
time_feature = pickle.load(open('data/final_time_feature.pkl', 'rb'))

# Initialize training neighbor finder to retrieve temporal graph
train_ngh_finder = get_neighbor_finder(train_data, args.uniform)

# Initialize validation and test neighbor finder to retrieve temporal graph
full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

# Initialize negative samplers. Set seeds for validation and testing so negatives are the same across different runs
# NB: in the inductive setting, negatives are sampled only amongst other new nodes

# train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations) # src_list, dst_list, portfolio_list, idx_list
# val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, full_data.portfolios, full_data.edge_idxs, seed=0)
# nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations, new_node_val_data.portfolios, new_node_val_data.edge_idxs, seed=1)
# test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, full_data.portfolios, full_data.edge_idxs, seed=2)
# nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources, new_node_test_data.destinations, new_node_test_data.portfolios, new_node_test_data.edge_idxs, seed=3)

# Set device
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

# Compute time statistics
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
  compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

for i in range(args.n_runs):
  results_path = "results/{}_{}.pkl".format(args.prefix, i) if i > 0 else "results/{}.pkl".format(args.prefix)
  Path("results/").mkdir(parents=True, exist_ok=True)

  # Initialize Model
  tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
            edge_features=edge_features, device=device,
            n_layers=NUM_LAYER,
            n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
            message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
            memory_update_at_start=not args.memory_update_at_end,
            embedding_module_type=args.embedding_module,
            message_function=args.message_function,
            aggregator_type=args.aggregator,
            memory_updater_type=args.memory_updater,
            n_neighbors=NUM_NEIGHBORS,
            mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
            mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
            use_destination_embedding_in_message=args.use_destination_embedding_in_message,
            use_source_embedding_in_message=args.use_source_embedding_in_message,
            dyrep=args.dyrep)
  optimizer = torch.optim.Adam(tgn.parameters(), lr=LEARNING_RATE)
  tgn = tgn.to(device)

  num_instance = len(train_data.sources)
  num_batch = math.ceil(num_instance / BATCH_SIZE)

  logger.info('num of training instances: {}'.format(num_instance))
  logger.info('num of batches per epoch: {}'.format(num_batch))
  idx_list = np.arange(num_instance)
  
  new_nodes_val_aps = []
  val_aps = []
  epoch_times = []
  total_epoch_times = []
  train_losses = []

  early_stopper = EarlyStopMonitor(max_round=args.patience)

  for epoch in range(NUM_EPOCH):
    start_epoch = time.time()

    """
    Training
    """

    # Reinitialize memory of the model at the start of each epoch
    if USE_MEMORY:
      tgn.memory.__init_memory__()

    # Train using only training graph
    tgn.set_neighbor_finder(train_ngh_finder)
    m_loss = []

    logger.info('start {} epoch'.format(epoch))
    for k in range(0, num_batch, args.backprop_every):
      loss = 0
      optimizer.zero_grad()

      # Custom loop to allow to perform backpropagation only every a certain number of batches
      for j in range(args.backprop_every):
        batch_idx = k + j

        if batch_idx >= num_batch:
          continue

        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(num_instance, start_idx + BATCH_SIZE)
        
        sources_batch = train_data.sources[start_idx:end_idx] # <class 'numpy.ndarray'> (BATCH_SIZE,)
        destinations_batch = train_data.destinations[start_idx:end_idx]
        edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
        timestamps_batch = train_data.timestamps[start_idx:end_idx]
        portfolios_batch = train_data.portfolios[start_idx:end_idx]
        
        # negative sampling
        train_rand_sampler = RandEdgeSampler(sources_batch, destinations_batch, portfolios_batch, edge_idxs_batch)
        size = 3 
        negatives_batch = train_rand_sampler.sample(size) 

        # potential pos sampling
        potential_pos_batch = []
        negative_batch_final = []
        ########### 1. negatives_batch와 timestamps_batch를 참고해서 negatives sample의 item feature 불러오기
        negatives_feature = []
        for n, t, port in zip(negatives_batch, timestamps_batch, portfolios_batch):
          neg_time_feature = time_feature[t][n]
          # port = port[1:-1].split(',')
          # port = list(map(int, port))
          portfolio_time_feature = time_feature[t][port]

          ########### 2. portfolios_batch에다가 negatives_batch 한개씩 붙이면서 새로운 portfolio 만들기
          sharpes = {}
          potential_pos = []
          for neg_index, neg in zip(n, neg_time_feature):
            new_portfolio_time_feature = np.append(portfolio_time_feature, neg.reshape(1,30), axis=0) # (n_portfolio+1, 30)

            ########### 3. 새로운 portfolio의 각각의 sharpe and return 구하기
            mu = np.mean(new_portfolio_time_feature, axis=1)
            Sigma = np.cov(new_portfolio_time_feature)
            try:
              Sigma_inv = np.linalg.inv(Sigma)
            except:
              print('Sigma is not invertible!')
              print('new_portfolio_time_feature')
              print(new_portfolio_time_feature)
              print('Sigma')
              print(Sigma)
            max_sharpe = np.sqrt(np.dot(mu, Sigma_inv).dot(mu.T))
            sharpes[neg_index] = max_sharpe
         
          ########### 4. 제일 좋은 sharpe을 보인 negatives_batch를 potential_pos로 두기
          potential_pos = sorted(sharpes.items(), key=lambda x: x[1], reverse=True)[:1][0][0]
          potential_pos_batch.append(potential_pos)

          potential_negs = [x[0] for x in sorted(sharpes.items(), key=lambda x: x[1], reverse=True)[1:]]
          negative_batch_final.append(potential_negs)

        ########### 5. sources_batch를 duplicate해서 negative_batch의 길이만큼 
        sources_batch_duplicated = [x for x in sources_batch for _ in range(size-1)]
        destinations_batch_duplicated = [x for x in destinations_batch for _ in range(size-1)]
        potential_pos_batch_duplicated = [x for x in potential_pos_batch for _ in range(size-1)]
        timestamps_batch_duplicated = [x for x in timestamps_batch for _ in range(size-1)]
        edge_idxs_batch_duplicated = [x for x in edge_idxs_batch for _ in range(size-1)]
        potential_neg_batch_duplicated = [x for y in negative_batch_final for x in y]

        ########### 6. BPR용 데이터 + CL용 데이터
        # <class 'numpy.ndarray'> (200,)
        sources_batch = np.array(sources_batch_duplicated + sources_batch_duplicated) 
        destinations_batch = np.array(destinations_batch_duplicated + potential_pos_batch_duplicated)
        negatives_batch = np.array(potential_neg_batch_duplicated + potential_neg_batch_duplicated)
        timestamps_batch = np.array(timestamps_batch_duplicated + timestamps_batch_duplicated)
        edge_idxs_batch = np.array(edge_idxs_batch_duplicated + edge_idxs_batch_duplicated)

        """
        emb 계산
        """
        tgn = tgn.train()
        # # torch.Size([800, 31]) # 800개 = BPR용 데이터 + CL용 데이터
        source_embedding, destination_embedding, negative_embedding = tgn.compute_temporal_embeddings(sources_batch,
                                                                                                    destinations_batch,
                                                                                                    negatives_batch,
                                                                                                    timestamps_batch,
                                                                                                    edge_idxs_batch,
                                                                                                    NUM_NEIGHBORS)
        
        
        """
        loss 계산
        """

        # get the first half
        source_embedding_BPR = source_embedding[:int(source_embedding.shape[0]/2)]
        destination_embedding_BPR = destination_embedding[:int(destination_embedding.shape[0]/2)]
        negative_embedding_BPR = negative_embedding[:int(negative_embedding.shape[0]/2)]

        pos_scores = torch.sum(source_embedding_BPR * destination_embedding_BPR, dim=1)
        neg_scores = torch.sum(source_embedding_BPR * negative_embedding_BPR, dim=1)
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))

        # get the second half
        source_embedding_CL = source_embedding[int(source_embedding.shape[0]/2):]
        destination_embedding_CL = destination_embedding[int(destination_embedding.shape[0]/2):]
        negative_embedding_CL = negative_embedding[int(negative_embedding.shape[0]/2):]     
        tau = 0.1
        pos_scores = torch.sum(torch.exp(source_embedding_CL * destination_embedding_CL)/tau, dim=1)
        neg_scores = torch.sum(torch.exp(source_embedding_CL * negative_embedding_CL)/tau, dim=1)
       
        cl_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        cl_loss = 0

        loss += bpr_loss
        loss += cl_loss
        
      loss /= args.backprop_every

      loss.backward()
      optimizer.step()
      m_loss.append(loss.item())

      # Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to
      # the start of time
      if USE_MEMORY:
        tgn.memory.detach_memory()

    epoch_time = time.time() - start_epoch
    epoch_times.append(epoch_time)

    """
    Validation
    """
    
    # Validation uses the full graph
    tgn.set_neighbor_finder(full_ngh_finder)

    if USE_MEMORY:
      # Backup memory at the end of training, so later we can restore it and use it for the
      # validation on unseen nodes
      train_memory_backup = tgn.memory.backup_memory()


    val_ap, val_auc, val_SR, val_New_SR, val_returns, val_New_returns = eval_recommendation(tgn=tgn,
                                                                                            data=val_data, 
                                                                                            batch_size=BATCH_SIZE,
                                                                                            n_neighbors=NUM_NEIGHBORS)

    if USE_MEMORY:
      val_memory_backup = tgn.memory.backup_memory()
      # Restore memory we had at the end of training to be used when validating on new nodes.
      # Also backup memory after validation so it can be used for testing (since test edges are
      # strictly later in time than validation edges)
      tgn.memory.restore_memory(train_memory_backup)

    # Validate on unseen nodes
    nn_val_ap, nn_val_auc, nn_val_SR, nn_val_New_SR, nn_val_returns, nn_val_New_returns = eval_recommendation(tgn,
                                                                                                              new_node_val_data, 
                                                                                                              BATCH_SIZE,
                                                                                                              n_neighbors=NUM_NEIGHBORS)

    if USE_MEMORY:
      # Restore memory we had at the end of validation
      tgn.memory.restore_memory(val_memory_backup)

    new_nodes_val_aps.append(nn_val_ap)
    val_aps.append(val_ap)
    train_losses.append(np.mean(m_loss))

    # Save temporary results to disk
    pickle.dump({
      "val_aps": val_aps,
      "new_nodes_val_aps": new_nodes_val_aps,
      "train_losses": train_losses,
      "epoch_times": epoch_times,
      "total_epoch_times": total_epoch_times
    }, open(results_path, "wb"))

    total_epoch_time = time.time() - start_epoch
    total_epoch_times.append(total_epoch_time)

    logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
    logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
    logger.info(
      'val auc: {}, new node val auc: {}'.format(val_auc, nn_val_auc))
    logger.info(
      'val ap: {}, new node val ap: {}'.format(val_ap, nn_val_ap))
    logger.info(
      'val SR: {}, new node val SR: {}'.format(val_SR, nn_val_SR))
    logger.info(
      'val New SR: {}, new node val New SR: {}'.format(val_New_SR, nn_val_New_SR))
    logger.info(
      'val returns: {}, new node val returns: {}'.format(val_returns, nn_val_returns))
    logger.info(
      'val New returns: {}, new node val New returns: {}'.format(val_New_returns, nn_val_New_returns))
    

    # Early stopping
    if early_stopper.early_stop_check(np.mean(val_ap)):
      logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
      logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
      best_model_path = get_checkpoint_path(early_stopper.best_epoch)
      tgn.load_state_dict(torch.load(best_model_path))
      logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
      tgn.eval()
      break
    else:
      torch.save(tgn.state_dict(), get_checkpoint_path(epoch))

  # Training has finished, we have loaded the best model, and we want to backup its current
  # memory (which has seen validation edges) so that it can also be used when testing on unseen
  # nodes
  if USE_MEMORY:
    val_memory_backup = tgn.memory.backup_memory()

  """
  Test
  """
  tgn.embedding_module.neighbor_finder = full_ngh_finder
  test_ap, test_auc, test_SR, test_New_SR, test_returns, test_New_returns = eval_recommendation(tgn,
                                                                                                test_data, 
                                                                                                BATCH_SIZE,
                                                                                                n_neighbors=NUM_NEIGHBORS)

  if USE_MEMORY:
    tgn.memory.restore_memory(val_memory_backup)

  # Test on unseen nodes
  nn_test_ap, nn_test_auc, nn_test_SR, nn_test_New_SR, nn_test_returns, nn_test_New_returns  = eval_recommendation(tgn,
                                                                                                                  new_node_test_data, 
                                                                                                                  BATCH_SIZE,
                                                                                                                  n_neighbors=NUM_NEIGHBORS)

  logger.info(
    'Test statistics: Old nodes -- ap: {}'.format(test_ap))
  logger.info(
    'Test statistics: New nodes -- ap: {}'.format(nn_test_ap))
  logger.info(
    'Test statistics: Old nodes -- auc: {}'.format(test_auc))
  logger.info(
    'Test statistics: New nodes -- auc: {}'.format(nn_test_auc))
  logger.info(
    'Test statistics: Old nodes -- SR: {}'.format(test_SR))
  logger.info(
    'Test statistics: New nodes -- SR: {}'.format(nn_test_SR))
  logger.info(
    'Test statistics: Old nodes -- New_SR: {}'.format(test_New_SR))
  logger.info(
    'Test statistics: New nodes -- New_SR: {}'.format(nn_test_New_SR))
  logger.info(
    'Test statistics: Old nodes -- returns: {}'.format(test_returns))
  logger.info(
    'Test statistics: New nodes -- returns: {}'.format(nn_test_returns))
  logger.info(
    'Test statistics: Old nodes -- New_returns: {}'.format(test_New_returns))
  logger.info(
    'Test statistics: New nodes -- New_returns: {}'.format(nn_test_New_returns))
  

  # Save results for this run
  pickle.dump({
    "val_aps": val_aps,
    "new_nodes_val_aps": new_nodes_val_aps,
    "test_ap": test_ap,
    "new_node_test_ap": nn_test_ap,
    "epoch_times": epoch_times,
    "train_losses": train_losses,
    "total_epoch_times": total_epoch_times
  }, open(results_path, "wb"))

  logger.info('Saving TGN model')
  if USE_MEMORY:
    # Restore memory at the end of validation (save a model which is ready for testing)
    tgn.memory.restore_memory(val_memory_backup)
  torch.save(tgn.state_dict(), MODEL_SAVE_PATH)
  logger.info('TGN model saved')
