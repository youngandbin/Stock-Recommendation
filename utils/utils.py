import numpy as np
import torch


class MergeLayer(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3, dim4):
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
    self.fc2 = torch.nn.Linear(dim3, dim4)
    self.act = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2], dim=1)
    h = self.act(self.fc1(x))
    return self.fc2(h)


class MLP(torch.nn.Module):
  def __init__(self, dim, drop=0.3):
    super().__init__()
    self.fc_1 = torch.nn.Linear(dim, 80)
    self.fc_2 = torch.nn.Linear(80, 10)
    self.fc_3 = torch.nn.Linear(10, 1)
    self.act = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(p=drop, inplace=False)

  def forward(self, x):
    x = self.act(self.fc_1(x))
    x = self.dropout(x)
    x = self.act(self.fc_2(x))
    x = self.dropout(x)
    return self.fc_3(x).squeeze(dim=1)


class EarlyStopMonitor(object):
  def __init__(self, max_round=3, higher_better=True, tolerance=1e-10):
    self.max_round = max_round
    self.num_round = 0

    self.epoch_count = 0
    self.best_epoch = 0

    self.last_best = None
    self.higher_better = higher_better
    self.tolerance = tolerance

  def early_stop_check(self, curr_val):
    if not self.higher_better:
      curr_val *= -1
    if self.last_best is None:
      self.last_best = curr_val
    elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
      self.last_best = curr_val
      self.num_round = 0
      self.best_epoch = self.epoch_count
    else:
      self.num_round += 1

    self.epoch_count += 1

    return self.num_round >= self.max_round


class RandEdgeSampler(object):
  def __init__(self, src_list, dst_list, portfolio_list, idx_list, seed=None):
    self.seed = None
    self.src_list = src_list
    self.dst_list = np.unique(dst_list)
    self.portfolio_list = portfolio_list
    self.idx_list = idx_list

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  # def sample(self, size):
  #   if self.seed is None:
  #     src_index = np.random.randint(0, len(self.src_list), size) # 0~len(self.src_list) 사이의 정수 중 size개를 랜덤하게 뽑음
  #     dst_index = np.random.randint(0, len(self.dst_list), size)
  #   else:
  #     src_index = self.random_state.randint(0, len(self.src_list), size)
  #     dst_index = self.random_state.randint(0, len(self.dst_list), size)
  #   return self.src_list[src_index], self.dst_list[dst_index]
  
  def sample(self, size):
    if self.seed is None:
      dst_index = []
      for i, src in enumerate(self.src_list): # 유저 한 명씩 돌면서~
        # dst_list 내에서 portfolio_list[i]에 해당하는 노드 빼고 랜덤으로 3개를 추출한다, 단, 비복원추출한다. 중복되서 뽑히는 게 없도록 한다.
        sample_dst = np.random.choice(np.setdiff1d(self.dst_list, self.portfolio_list[i]), size=3, replace=False)
        dst_index.append(sample_dst)
    else:
      pass
    return np.array(dst_index) 
        
      

  def reset_random_state(self):
    self.random_state = np.random.RandomState(self.seed)


def get_neighbor_finder(data, uniform, max_node_idx=None):
  max_node_idx = max(data.sources.max(), data.destinations.max()) if max_node_idx is None else max_node_idx
  adj_list = [[] for _ in range(max_node_idx + 1)]
  for source, destination, edge_idx, timestamp in zip(data.sources, data.destinations,
                                                      data.edge_idxs,
                                                      data.timestamps):
    adj_list[source].append((destination, edge_idx, timestamp)) # 예를 들어 1번 노드의 이웃이 3개 있다면, adj_list[1] = [(2, 0, 0), (3, 1, 1), (4, 2, 2)] 이런 식으로 저장됨
    adj_list[destination].append((source, edge_idx, timestamp)) # 예를 들어 10번 노드의 이웃이 2개 있다면, adj_list[10] = [(9, 3, 3), (11, 4, 4)] 이런 식으로 저장됨

  return NeighborFinder(adj_list, uniform=uniform)


class NeighborFinder:
  def __init__(self, adj_list, uniform=False, seed=None):
    self.node_to_neighbors = []
    self.node_to_edge_idxs = []
    self.node_to_edge_timestamps = []

    for neighbors in adj_list:
      # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
      # We sort the list based on timestamp
      sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
      self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors])) # [neighbor1, neighbor2, ...] # 위의 예시에서는 [2, 3, 4]가 됨
      self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighhbors])) # [edge_idx1, edge_idx2, ...] # 위의 예시에서는 [0, 1, 2]가 됨
      self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighhbors])) # [timestamp1, timestamp2, ...] # 위의 예시에서는 [0, 1, 2]가 됨

    self.uniform = uniform

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def find_before(self, src_idx, cut_time):
    """
    Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph. The returned interactions are sorted by time.

    Returns 3 lists: neighbors, edge_idxs, timestamps

    """
    # 기준 ts의 위치(index)를 찾는다
    i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)

    # 첫 번째 반환값: src_idx 노드의 이웃 indices 중에서 [:i]만큼만 가져온다
    return self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], self.node_to_edge_timestamps[src_idx][:i]

  def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
    """
    Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

    Params
    ------
    src_idx_l: List[int]
    cut_time_l: List[float],
    num_neighbors: int
    """
    assert (len(source_nodes) == len(timestamps))

    tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
    # NB! All interactions described in these matrices are sorted in each row by time
    neighbors = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
    edge_times = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.float32)  # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
    edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]

    # 노드 하나씩 돌면서~
    for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):

      # 여기서 neighbors를 찾는다~
      # train data의 source, destination 저장된 adj_list를 가지고 이웃을 판단한다. 
      # 그래서 negative의 경우 destination node를 기준으로 이웃을 판단하게 된다.
      # extracts all neighbors, interactions indexes and timestamps of all interactions of user source_node happening before cut_time
      source_neighbors, source_edge_idxs, source_edge_times = self.find_before(source_node, timestamp)  

      if len(source_neighbors) > 0 and n_neighbors > 0:
        if self.uniform:  # if we are applying uniform sampling, shuffles the data above before sampling
          sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors)

          neighbors[i, :] = source_neighbors[sampled_idx]
          edge_times[i, :] = source_edge_times[sampled_idx]
          edge_idxs[i, :] = source_edge_idxs[sampled_idx]

          # re-sort based on time
          pos = edge_times[i, :].argsort()
          neighbors[i, :] = neighbors[i, :][pos]
          edge_times[i, :] = edge_times[i, :][pos]
          edge_idxs[i, :] = edge_idxs[i, :][pos]
        else:
          # Take most recent interactions
          source_edge_times = source_edge_times[-n_neighbors:]
          source_neighbors = source_neighbors[-n_neighbors:]
          source_edge_idxs = source_edge_idxs[-n_neighbors:]

          assert (len(source_neighbors) <= n_neighbors)
          assert (len(source_edge_times) <= n_neighbors)
          assert (len(source_edge_idxs) <= n_neighbors)

          neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors
          edge_times[i, n_neighbors - len(source_edge_times):] = source_edge_times
          edge_idxs[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs

    return neighbors, edge_idxs, edge_times