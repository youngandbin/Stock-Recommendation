import math
import random
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def eval_edge_prediction(model, negative_edge_sampler, data, n_neighbors, batch_size=200):
  # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
  # negatives for validation / test set)
  assert negative_edge_sampler.seed is not None
  negative_edge_sampler.reset_random_state()

  val_ap, val_auc = [], []
  with torch.no_grad():
    model = model.eval()
    # While usually the test batch size is as big as it fits in memory, here we keep it the same
    # size as the training batch size, since it allows the memory to be updated more frequently,
    # and later test batches to access information from interactions in previous test batches
    # through the memory
    TEST_BATCH_SIZE = batch_size
    num_test_instance = len(data.sources)
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

    for k in range(num_test_batch):
      s_idx = k * TEST_BATCH_SIZE
      e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
      sources_batch = data.sources[s_idx:e_idx]
      destinations_batch = data.destinations[s_idx:e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = data.edge_idxs[s_idx: e_idx]

      size = len(sources_batch)
      _, negative_samples = negative_edge_sampler.sample(size)

      pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, 
                                                            destinations_batch,
                                                            negative_samples, 
                                                            timestamps_batch,
                                                            edge_idxs_batch, 
                                                            n_neighbors)

      pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
      true_label = np.concatenate([np.ones(size), np.zeros(size)])

      val_ap.append(average_precision_score(true_label, pred_score))
      val_auc.append(roc_auc_score(true_label, pred_score))

  return np.mean(val_ap), np.mean(val_auc)


def eval_node_classification(tgn, decoder, data, edge_idxs, batch_size, n_neighbors):
  pred_prob = np.zeros(len(data.sources))
  num_instance = len(data.sources)
  num_batch = math.ceil(num_instance / batch_size)

  with torch.no_grad():
    decoder.eval()
    tgn.eval()
    for k in range(num_batch):
      s_idx = k * batch_size
      e_idx = min(num_instance, s_idx + batch_size)

      sources_batch = data.sources[s_idx: e_idx]
      destinations_batch = data.destinations[s_idx: e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = edge_idxs[s_idx: e_idx]

      source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,
                                                                                   destinations_batch,
                                                                                   destinations_batch,
                                                                                   timestamps_batch,
                                                                                   edge_idxs_batch,
                                                                                   n_neighbors)
      pred_prob_batch = decoder(source_embedding).sigmoid()
      pred_prob[s_idx: e_idx] = pred_prob_batch.cpu().numpy()

  auc_roc = roc_auc_score(data.labels, pred_prob)
  return auc_roc


def recall_at_k(recommendations, test_items, k):
    hits = len(set(recommendations[:k]) & set(test_items))
    return hits / min(k, len(test_items))

def ndcg_at_k(recommendations, test_items, k):
    dcg = 0
    idcg = sum([1 / np.log2(i + 2) for i in range(min(k, len(test_items)))])
    for i, item in enumerate(recommendations[:k]):
        if item in test_items:
            dcg += 1 / np.log2(i + 2)
    return dcg / idcg

def eval_recommendation(tgn, negative_edge_sampler, data, batch_size, n_neighbors):

  with torch.no_grad():
    tgn.eval()
  
    source_nodes = data.sources
    destination_nodes = data.destinations
    size = len(source_nodes)
    _, negative_nodes = negative_edge_sampler.sample(size)
    edge_times = data.timestamps
    edge_idxs = data.edge_idxs
    
    """
    node embedding 생성
    """
    source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(source_nodes,
                                                                                  destination_nodes,
                                                                                  destination_nodes,
                                                                                  edge_times,
                                                                                  edge_idxs,
                                                                                  n_neighbors)
    
    """
    유저마다 user_purchase_history 생성
    """
    
    # create a dict 'user_buy_dict' where keys are unique source_nodes and values are lists of destination_nodes that the source_nodes have purchased
    source_nodes_set = np.unique(source_nodes)
    destination_nodes_set = np.unique(destination_nodes)
    user_buy_dict = {source_node: destination_nodes[source_nodes == source_node] for source_node in source_nodes_set}
    
    """
    유저 loop 돌면서 평가
    """
    
    # print('Evaluation Start, num of users: ', len(user_buy_dict), len(source_nodes_set))
    # print('Evaluation Start, num of items: ', len(destination_nodes_set))
    sum_recall = 0.0
    sum_ndcg = 0.0
    total_user = 0

    for user, pos_items in user_buy_dict.items():
      
      """
      예시
      user:  1                                                      # numpy.int64
      pos_items:  [274 274 274 274 274 274 274 274 274 274 274 274] # numpy.ndarray
      neg_items = [517]                                             # numpy.ndarray
      """
      
      # pos_items 없는 유저는 평가에서 제외
      if len(pos_items) == 0:
        continue
      
      neg_items = np.setdiff1d(destination_nodes_set, pos_items)
      # neg_items 100개 미만인 유저는 평가에서 제외
      if len(neg_items) < 100:
        continue
      neg_items = random.sample(list(neg_items), 100)
      
      user_tensor = torch.LongTensor([user]).to(tgn.device)
      pos_tensor = torch.LongTensor(pos_items).to(tgn.device)
      neg_tensor = torch.LongTensor(neg_items).to(tgn.device)
      
      user_emb = source_embedding[user_tensor]
      pos_emb = destination_embedding[pos_tensor]
      neg_emb = destination_embedding[neg_tensor]
      
      pos_scores = torch.sum(user_emb * pos_emb, dim=1)
      neg_scores = torch.sum(user_emb * neg_emb, dim=1)
      
      # 예진 
      k = 10
      ranking = torch.argsort(torch.cat([pos_scores.flatten(), neg_scores.flatten()]), descending=True).cpu().numpy().tolist()
      pos_ranking = [i for i in range(len(pos_scores))]

      recall = recall_at_k(ranking, pos_ranking, k)
      ndcg = ndcg_at_k(ranking, pos_ranking, k)

      sum_recall += recall   
      sum_ndcg += ndcg

      total_user += 1

    ap = sum_recall/total_user
    auc = sum_ndcg/total_user

    return ap, auc