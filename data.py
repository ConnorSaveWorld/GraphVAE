
import random

import networkx as nx
import numpy as np
import torch
from scipy.sparse import *
from  Synthatic_graph_generator import *
# from util import *
import os
import pickle as pkl
import scipy.sparse as sp
import warnings
import pandas as pd
import dgl as dgl

import ogb
import pickle

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

# load cora, citeseer and pubmed dataset
def Graph_load(dataset = 'cora'):
    '''
    Load a single graph dataset
    :param dataset: dataset name
    :return:
    '''
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        load = pkl.load(open("data/Kernel_dataset/ind.{}.{}".format(dataset, names[i]), 'rb'), encoding='latin1')
        # print('loaded')
        objects.append(load)
        # print(load)
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/Kernel_dataset/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    G = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(G)
    return adj, features, G

def graph_load_batch(data_dir,
                     min_num_nodes=20,
                     max_num_nodes=1000,
                     name='ENZYMES',
                     node_attributes=True,
                     graph_labels=True):
  '''
    load many graphs, e.g. enzymes
    :return: a list of graphs
    '''
  print('Loading graph dataset: ' + str(name))
  G = nx.Graph()
  # load data
  path = os.path.join(data_dir, name)
  data_adj = np.loadtxt(
      os.path.join(path, '{}_A.txt'.format(name)), delimiter=',').astype(int)
  if node_attributes:
    data_node_att = np.loadtxt(
        os.path.join(path, '{}_node_attributes.txt'.format(name)),
        delimiter=',')
  data_node_label = np.loadtxt(
      os.path.join(path, '{}_node_labels.txt'.format(name)),
      delimiter=',').astype(int)
  data_graph_indicator = np.loadtxt(
      os.path.join(path, '{}_graph_indicator.txt'.format(name)),
      delimiter=',').astype(int)
  if graph_labels:
    data_graph_labels = np.loadtxt(
        os.path.join(path, '{}_graph_labels.txt'.format(name)),
        delimiter=',').astype(int)

  data_tuple = list(map(tuple, data_adj))
  # print(len(data_tuple))
  # print(data_tuple[0])

  # add edges
  G.add_edges_from(data_tuple)
  # add node attributes
  for i in range(data_node_label.shape[0]):
    if node_attributes:
      G.add_node(i + 1, feature=data_node_att[i])
    G.add_node(i + 1, label=data_node_label[i])
  G.remove_nodes_from(list(nx.isolates(G)))

  # remove self-loop
  G.remove_edges_from(nx.selfloop_edges(G))

  # print(G.number_of_nodes())
  # print(G.number_of_edges())

  # split into graphs
  graph_num = data_graph_indicator.max()
  node_list = np.arange(data_graph_indicator.shape[0]) + 1
  graphs = []
  max_nodes = 0
  for i in range(graph_num):
    # find the nodes for each graph
    nodes = node_list[data_graph_indicator == i + 1]
    G_sub = G.subgraph(nodes)
    G_sub = nx.Graph((G_sub))
    if graph_labels:
      G_sub.graph['label'] = data_graph_labels[i]
    # print('nodes', G_sub.number_of_nodes())
    # print('edges', G_sub.number_of_edges())
    # print('label', G_sub.graph)
    if G_sub.number_of_nodes() >= min_num_nodes and G_sub.number_of_nodes(
    ) <= max_num_nodes:
      graphs.append(G_sub)
      if G_sub.number_of_nodes() > max_nodes:
        max_nodes = G_sub.number_of_nodes()
      # print(G_sub.number_of_nodes(), 'i', i)
      # print('Graph dataset name: {}, total graph num: {}'.format(name, len(graphs)))
      # logging.warning('Graphs loaded, total num: {}'.format(len(graphs)))
  print('Loaded')
  list_adj = []
  list_x= []
  list_label = []
  for G in graphs:
      list_adj.append(nx.adjacency_matrix(G))
      list_x.append(None)
      list_label.append(G.graph['label']-1)
  return list_adj, list_x, list_label


class Datasets():
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_adjs,self_for_none, list_Xs, graphlabels = None, padding =True, Max_num = None, set_diag_of_isol_Zer=True):
        """
        :param list_adjs: a list of adjacency in sparse format
        :param list_Xs: a list of node feature matrix
        :param graphlabels: a list of int, that indicate correponding class of element in list_adjs

        """
        'Initialization'
        self.num_views = len(list_adjs[0]) if list_adjs and isinstance(list_adjs[0], list) else 1
        
        if self.num_views > 1:
            print(f"Dataset initialized with {self.num_views} views per graph.")


        if Max_num!=0 and Max_num!=None:
            list_adjs, graphlabels, list_Xs = self.remove_largergraphs( list_adjs, graphlabels, list_Xs, Max_num)
        self.set_diag_of_isol_Zer = set_diag_of_isol_Zer
        self.paading = padding
        self.list_Xs = list_Xs
        self.labels = graphlabels
        self.list_adjs = list_adjs
        self.toatl_num_of_edges = 0
        self.max_num_nodes = 0
        for i, multi_view_adj in enumerate(list_adjs):
            current_views = multi_view_adj if self.num_views > 1 else [multi_view_adj]
            if self.max_num_nodes < current_views[0].shape[0]:
                self.max_num_nodes = current_views[0].shape[0]

            for v_idx in range(len(current_views)):
                adj = current_views[v_idx]
                adj =  adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
                adj += sp.eye(adj.shape[0])
                self.toatl_num_of_edges += adj.sum().item()
                if self.num_views > 1:
                    list_adjs[i][v_idx] = adj
                else:
                    list_adjs[i] = adj
            # list_adjs[i] =  adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
            # list_adjs[i] += sp.eye(list_adjs[i].shape[0])
            # if self.max_num_nodes < adj.shape[0]:
            #     self.max_num_nodes = adj.shape[0]
            # self.toatl_num_of_edges += adj.sum().item() #update


            # if list_Xs!=None:
            #     self.list_adjs[i], list_Xs[i] = self.permute(list_adjs[i], list_Xs[i])
            # else:
            #     self.list_adjs[i], _ = self.permute(list_adjs[i], None)
        # if Max_num!=None:
        #     self.max_num_nodes = Max_num
        # self.processed_Xs = []
        # self.processed_adjs = []
        # self.num_of_edges = []
        # # for i in range(len(self.list_Xs)):
        # for i in range(self.__len__()):
        #     a,x,n,_ = self.process(i,self_for_none)
        #     self.processed_Xs.append(x)
        #     self.processed_adjs.append(a)
        #     self.num_of_edges.append(n)
        self.feature_size = 0 
        # self.feature_size = self.processed_Xs[0].shape[-1]
        self.adj_s= []
        self.x_s = []
        self.num_nodes = []
        self.subgraph_indexes = []

        self.featureList = None

        

  def remove_largergraphs(self, adjs, labels, Xs, max_size):
      processed_adjs = []
      processed_labels = []
      processed_Xs=[]

      if not adjs: # Handle empty list
        print("adjs is empty")
        return [], [], []
      is_multi_view = isinstance(adjs[0], list)

      for i in range(len(adjs)):
          if is_multi_view:
            # For multi-view graphs, check the shape of the first view
            shape_to_check = adjs[i][0].shape
          else:
            # For single-view graphs, check the shape directly
            shape_to_check = adjs[i].shape

          if shape_to_check[0] <= max_size:
            processed_adjs.append(adjs[i])
            if labels is not None:
                processed_labels.append(labels[i])
            if Xs is not None:
                processed_Xs.append(Xs[i])
                
    # Handle the case where the labels/features list might have been None initially
      final_labels = processed_labels if labels is not None else None
      final_Xs = processed_Xs if Xs is not None else None
    
      return processed_adjs, final_labels, final_Xs
  def get(self):
      indexces = list(range(self.__len__()))
      return [self.processed_adjs[i] for i in indexces], [self.processed_Xs[i] for i in indexces]

  def set_features(self, some_feature, ):
      self.featureList = some_feature
      # self.labels = labels


  def get_adj_list(self):
      return self.adj_s

  def get__(self,from_, to_, self_for_none, bfs=None, ignore_isolate_nodes = False, get_labels=False):
    adj_s_batch = self.adj_s[from_:to_]
    x_s_batch = self.x_s[from_:to_]
    num_nodes_batch = self.num_nodes[from_:to_]
    subgraph_indexes_batch = self.subgraph_indexes[from_:to_]

    # Handle the kernel features (target_kelrnel_val)
    # Only process this if featureList is not None.
    graphfeatures_batch = []
    if self.featureList is not None:
        for element in self.featureList:
            # Assuming element is a list/array of features for all graphs
            graphfeatures_batch.append(element[from_:to_])
    
    if get_labels:
        if self.labels is not None:
            labels_batch = self.labels[from_:to_]
            # The original return statement had 5 items. We now add labels as the 6th.
            return adj_s_batch, x_s_batch, num_nodes_batch, subgraph_indexes_batch, graphfeatures_batch, labels_batch
        else:
            # Handle the case where labels were not provided to the dataset
            # We'll return None for the labels.
            return adj_s_batch, x_s_batch, num_nodes_batch, subgraph_indexes_batch, graphfeatures_batch, None
    else:
        # If get_labels is False, maintain the original behavior
        return adj_s_batch, x_s_batch, num_nodes_batch, subgraph_indexes_batch, graphfeatures_batch



  def get_max_degree(self):
      return np.max([adj.sum(-1) for adj in self.processed_adjs])
  def processALL(self, self_for_none, bfs=None, ignore_isolate_nodes = False):
      self.adj_s = []
      self.x_s = []
      self.num_nodes = []
      self.subgraph_indexes = []
      # padded_to = max([self.list_adjs[i].shape[1] for i in range(from_, to_)])
      # padded_to = 225

      if not self.list_adjs: # Handle empty dataset
        print("list_adjs is null")
        return

      for i in range(len(self.list_adjs)):
        adj_views_tensor, x, num_node, indexes = self.process(
            i, self_for_none, None, bfs, ignore_isolate_nodes
        )
        self.adj_s.append(adj_views_tensor)
        self.x_s.append(x)
        self.num_nodes.append(num_node)
        self.subgraph_indexes.append(indexes)

      if self.x_s:
        self.feature_size = self.x_s[0].shape[-1]
      else:
        self.feature_size = 0

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_adjs)

  # In data.py, inside the Datasets class

  def process(self, index, self_for_none, padded_to=None, bfs_max_length=None, ignore_isolate_nodes=True):
    """
    Processes a single graph sample (either single or multi-view) from the raw data list.
    Returns padded tensor representations for adjacencies and features.
    """
    
    # --- Step 1: Standardize input to be a list of views ---
    is_multi_view = isinstance(self.list_adjs[index], list)
    adj_views_list = self.list_adjs[index] if is_multi_view else [self.list_adjs[index]]
    
    # --- Step 2: Determine graph and padding dimensions ---
    num_nodes = adj_views_list[0].shape[0] # Get shape from the first view

    if self.paading:
        max_num_nodes = self.max_num_nodes if padded_to is None else padded_to
    else:
        max_num_nodes = num_nodes

    # --- Step 3: Process Adjacency Matrices ---
    num_views = len(adj_views_list)
    padded_adj_views = torch.zeros(num_views, max_num_nodes, max_num_nodes)

    for v_idx in range(num_views):
        adj_view_sparse = adj_views_list[v_idx]
        adj_padded = lil_matrix((max_num_nodes, max_num_nodes))
        adj_padded[:num_nodes, :num_nodes] = adj_view_sparse[:, :]
        
        adj_padded.setdiag(0) # Remove existing self-loops

        if self_for_none:
            adj_padded.setdiag(1)
        else:
            adj_padded[:num_nodes, :num_nodes] += sp.eye(num_nodes)
            
        padded_adj_views[v_idx] = torch.tensor(adj_padded.toarray(), dtype=torch.float32)

    # If the original data was single-view, remove the unnecessary view dimension.
    if not is_multi_view:
        padded_adj_views = padded_adj_views.squeeze(0)

    # --- Step 4: Process Node Features ---
    if self.list_Xs and self.list_Xs[index] is not None:
        # If features are provided
        X_unpadded = self.list_Xs[index]
        feature_dim = X_unpadded.shape[1]
        X = np.zeros((max_num_nodes, feature_dim))
        X[:num_nodes, :] = X_unpadded
    else:
        # If no features, create default ones (identity + degree)
        first_view_padded = lil_matrix(padded_adj_views[0].numpy() if is_multi_view else padded_adj_views.numpy())
        
        identity_features = np.identity(max_num_nodes)
        degree_features = np.array(first_view_padded.sum(axis=1))
        
        # Normalize degree features
        if degree_features.sum() > 0:
            degree_features = degree_features / degree_features.sum()

        X = np.concatenate([identity_features, degree_features], axis=1)

    X = torch.tensor(X, dtype=torch.float32)

    # --- Step 5: Finalize and Return ---
    # The actual BFS permutation should happen *before* calling this method.
    # This method just needs to return the node indices being used.
    # For now, we assume the whole (padded) graph is used.
    final_indices = list(range(max_num_nodes))

    return padded_adj_views, X, num_nodes, final_indices
  # def process(self,index,self_for_none, padded_to=None,):
  #
  #     num_nodes = self.list_adjs[index].shape[0]
  #     if self.paading == True:
  #         max_num_nodes = self.max_num_nodes if padded_to==None else padded_to
  #     else:
  #         max_num_nodes = num_nodes
  #     adj_padded = lil_matrix((max_num_nodes,max_num_nodes)) # make the size equal to maximum graph
  #     if max_num_nodes==num_nodes:
  #         adj_padded = lil_matrix(self.list_adjs[index], dtype=np.int8)
  #     else:
  #       adj_padded[:num_nodes, :num_nodes] = self.list_adjs[index][:, :]
  #     adj_padded -= sp.dia_matrix((adj_padded.diagonal()[np.newaxis, :], [0]), shape=adj_padded.shape)
  #     if self_for_none:
  #       adj_padded += sp.eye(max_num_nodes)
  #     else:
  #         if max_num_nodes != num_nodes:
  #             adj_padded[:num_nodes, :num_nodes] += sp.eye(num_nodes)
  #         else:
  #             adj_padded += sp.eye(num_nodes)
  #     # adj_padded+= sp.eye(max_num_nodes)
  #
  #
  #
  #
  #     if self.list_Xs == None:
  #         # if the feature is not exist we use identical matrix
  #         X = np.identity( max_num_nodes)
  #         node_degree = adj_padded.sum(0)
  #         X = np.concatenate((node_degree.transpose(), X),1 )
  #
  #     else:
  #         #ToDo: deal with data with diffrent number of nodes
  #         X = self.list_Xs[index].toarray()
  #
  #     # adj_padded, X = self.permute(adj_padded, X)
  #
  #     # Converting sparse matrix to sparse tensor
  #     coo = adj_padded.tocoo()
  #     values = coo.data
  #     indices = np.vstack((coo.row, coo.col))
  #     i = torch.LongTensor(indices)
  #     v = torch.FloatTensor(values)
  #     shape = coo.shape
  #     adj_padded = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()
  #     X = torch.tensor(X, dtype=torch.int8)
  #
  #     return adj_padded.reshape(1,*adj_padded.shape), X.reshape(1, *X.shape), num_nodes

  # def permute(self, list_adj, X):
  #           p = list(range(list_adj.shape[0]))
  #           np.random.shuffle(p)
  #           # for i in range(list_adj.shape[0]):
  #           #     list_adj[:, i] = list_adj[p, i]
  #           #     X[:, i] = X[p, i]
  #           # for i in range(list_adj.shape[0]):
  #           #     list_adj[i, :] = list_adj[i, p]
  #           #     X[i, :] = X[i, p]
  #           list_adj[:, :] = list_adj[p, :]
  #           list_adj[:, :] = list_adj[:, p]
  #           if X !=None:
  #               X[:, :] = X[p, :]
  #               X[:, :] = X[:, p]
  #           return list_adj , X

  def shuffle(self):


      indx = list(range(len(self.list_adjs)))
      np.random.shuffle(indx)

      if  self.list_Xs !=None:
        self.list_Xs=[self.list_Xs[i] for i in indx]
      else:
          warnings.warn("X is empty")

      self.list_adjs=[self.list_adjs[i] for i in indx]

      # if the graphs have extracted features
      if self.featureList !=None:
          for el_i , element in enumerate(self.featureList):
              self.featureList[el_i] = element[indx]
      else:
          warnings.warn("Graph structureal feature is an empty Set")

      if self.labels != None:
          self.labels= [self.labels[i] for i in indx]
      else:
           warnings.warn("Label is an empty Set")

      if len(self.subgraph_indexes)>0:
          self.adj_s= [self.adj_s[i] for i in indx]
          self.x_s = [self.x_s[i] for i in indx]
          self.num_nodes = [self.num_nodes[i] for i in indx]
          self.subgraph_indexes = [self.subgraph_indexes[i] for i in indx]


  def __getitem__(self, index):
        'Generates one sample of data'
        # return self.processed_adjs[index], self.processed_Xs[index],torch.tensor(self.list_adjs[index].todense(), dtype=torch.float32)
        return self.processed_adjs[index], self.processed_Xs[index]
# generate a list of graph
def list_graph_loader( graph_type, _max_list_size=None, return_labels=False, limited_to=None):
  list_adj = []
  list_x =[]
  list_labels = []

  if graph_type == "Multi":
    list_adj, list_x, list_labels = load_neuroimaging_data("/root/GraphVAE-MM/dataset/Multi")

  elif graph_type=="IMDBBINARY":
      data = dgl.data.GINDataset(name='IMDBBINARY', self_loop=False)
      graphs, labels = data.graphs, data.labels
      for i, graph in enumerate(graphs):
          list_adj.append(csr_matrix(graph.adjacency_matrix().to_dense().numpy()))
          # list_x.append(graph.ndata['feat'])
          list_x.append(None)
          list_labels.append(labels[i].cpu().item())
      graphs_to_writeOnDisk = [gr.toarray() for gr in list_adj]
      np.save('IMDBBINARY_lattice_graph.npy', graphs_to_writeOnDisk, allow_pickle=True)

  elif graph_type=="NCI1":
      data = dgl.data.GINDataset(name='NCI1', self_loop=False)
      graphs, labels = data.graphs, data.labels
      for i, graph in enumerate(graphs):
          list_adj.append(csr_matrix(graph.adjacency_matrix().to_dense().numpy()))
          # list_x.append(graph.ndata['feat'])
          list_x.append(None)
          list_labels.append(labels[i].cpu().item())
      graphs_to_writeOnDisk = [gr.toarray() for gr in list_adj]
      np.save('NCI1_lattice_graph.npy', graphs_to_writeOnDisk, allow_pickle=True)
  elif graph_type=="MUTAG":
      data = dgl.data.GINDataset(name='MUTAG', self_loop=False)
      graphs, labels = data.graphs, data.labels
      for i, graph in enumerate(graphs):
          list_adj.append(csr_matrix(graph.adjacency_matrix().to_dense().numpy()))
          # list_x.append(graph.ndata['feat'])
          list_x.append(None)
          list_labels.append(labels[i].cpu().item())
      graphs_to_writeOnDisk = [gr.toarray() for gr in list_adj]
      np.save('MUTAG_lattice_graph.npy', graphs_to_writeOnDisk, allow_pickle=True)
  elif graph_type=="COLLAB":
      data = dgl.data.GINDataset(name='COLLAB', self_loop=False)
      graphs, labels = data.graphs, data.labels
      for i, graph in enumerate(graphs):
          list_adj.append(csr_matrix(graph.adjacency_matrix().to_dense().numpy()))
          # list_x.append(graph.ndata['feat'])
          list_x.append(None)
          list_labels.append(labels[i].cpu().item())
      graphs_to_writeOnDisk = [gr.toarray() for gr in list_adj]
      # np.save('COLLAB_lattice_graph.npy', graphs_to_writeOnDisk, allow_pickle=True)
  elif graph_type=="PTC":
      data = dgl.data.GINDataset(name='PTC', self_loop=False)
      graphs, labels = data.graphs, data.labels
      for i, graph in enumerate(graphs):
          list_adj.append(csr_matrix(graph.adjacency_matrix().to_dense().numpy()))
          # list_x.append(graph.ndata['feat'])
          list_x.append(None)
          list_labels.append(labels[i].cpu().item())
      graphs_to_writeOnDisk = [gr.toarray() for gr in list_adj]
      np.save('PTC_lattice_graph.npy', graphs_to_writeOnDisk, allow_pickle=True)
  elif graph_type == "PROTEINS":
      data = dgl.data.GINDataset(name='PROTEINS', self_loop=False)
      graphs, labels = data.graphs, data.labels
      for i, graph in enumerate(graphs):
          if graph.adjacency_matrix().shape[0]<100:
              list_adj.append(csr_matrix(graph.adjacency_matrix().to_dense().numpy()))
              # list_x.append(graph.ndata['feat'])
              list_x.append(None)
              list_labels.append(labels[i].cpu().item())
      # graphs_to_writeOnDisk = [gr.toarray() for gr in list_adj]
      # np.save('PROTEINS.npy', graphs_to_writeOnDisk, allow_pickle=True)
  elif graph_type == "QM9":
      data = dgl.data.QM9Dataset(label_keys=['mu'])
      for i, graph in enumerate(data):
          # if i==1000:
          #     break
          adj = dgl.to_homo(graph[0]).adjacency_matrix().to_dense().numpy()
          list_adj.append(scipy.sparse.csr_matrix(adj))
          list_x.append(None)
          # list_labels.append(labels[i].cpu().item())

  elif graph_type=="ogbg-molbbbp":
      # https://ogb.stanford.edu/docs/graphprop/
      from ogb.graphproppred import DglGraphPropPredDataset, collate_dgl
      d_name = "ogbg-molbbbp"  # ogbg-molhiv   'ogbg-code2' ogbg-ppa
      dataset = DglGraphPropPredDataset(name=d_name)


      list_adj = []
      for graph, label in dataset:
          list_adj.append(csr_matrix(graph.adjacency_matrix().to_dense().numpy()))
          # list_x.append(graph.ndata['feat'])
          list_x.append(None)
          list_labels.append(label.cpu().item())

      # graphs_to_writeOnDisk = [gr.toarray() for gr in list_adj]
      # np.save('ogbg-molbbbp.npy', graphs_to_writeOnDisk, allow_pickle=True)


      # list_labels = [adj.sum() for adj in list_adj]
  elif graph_type=="large_grid":
      for i in range(10):
            list_adj.append(nx.adjacency_matrix(grid(30, 100)))
            list_x.append(None)
  elif graph_type=="grid":
      for i in range(10, 20):
        for j in range(10, 20):
            list_adj.append(nx.adjacency_matrix(grid(i, j)))
            list_x.append(None)

  elif graph_type=="triangular_grid":
      for i in range(10, 20):
        for j in range(10, 20):
            list_adj.append(nx.adjacency_matrix(nx.triangular_lattice_graph(i, j)))
            list_x.append(None)
      # graphs_to_writeOnDisk = [gr.toarray() for  gr in list_adj]
      # np.save('triangular_lattice_graph.npy', graphs_to_writeOnDisk, allow_pickle=True)
  elif graph_type=="small_triangular_grid":
      for i in range(6, 12):
        for j in range(6, 12):
            list_adj.append(nx.adjacency_matrix(nx.triangular_lattice_graph(i, j)))
            list_x.append(None)
      # graphs_to_writeOnDisk = [gr.toarray() for  gr in list_adj]
      # np.save('triangular_lattice_graph.npy', graphs_to_writeOnDisk, allow_pickle=True)
  elif graph_type=="fancy_grid":
      for i in range(4, 8):
        for j in range(4, 8):
            list_adj.append(nx.adjacency_matrix(grid(i, j)))
      list_adj = padd_adj_to(list_adj, np.max(np.array([adj.shape[0] for adj in list_adj])))
      for adj in list_adj:
        list_x.append(node_festure_creator(adj, 3,10))
  elif graph_type == "tree":
      for graph_size in range(3, 83):
          list_x.append(None)
          list_adj.append(nx.adjacency_matrix(nx.random_tree(graph_size)))

  elif graph_type == "star":
      for graph_size in range(3,83):
          list_x.append(None)
          list_adj.append(nx.adjacency_matrix(nx.star_graph(graph_size)))

  elif graph_type == "wheel_graph":
      for graph_size in range(3,83):
          list_x.append(None)
          list_adj.append(nx.adjacency_matrix(nx.wheel_graph(graph_size)))
  elif graph_type=="IMDbMulti":
      list_adj = pkl.load(open("data/IMDbMulti/IMDBMulti.p",'rb'))
      list_x= [None for x in list_adj]
  elif graph_type=="one_grid":
        list_adj.append(nx.adjacency_matrix(grid(350, 10)))
        list_x.append(None)
  elif graph_type=="small_grid":
      for i in range(2, 3):
        for j in range(2, 5):
            list_adj.append(nx.adjacency_matrix(grid(i, j)))
            list_x.append(None)
  elif graph_type=="huge_grids":
      for i in range(4, 10):
          for j in range(4, 10):
              list_adj.append(nx.adjacency_matrix(grid(i, j)))
              list_x.append(None)
  elif graph_type=="community":
      for i in range(30, 81):
        for j in range(30,81):
            list_adj.append(nx.adjacency_matrix(n_community([i, j], p_inter=0.3, p_intera=0.05)))
            list_x.append(None)

  elif graph_type=="multi_community":
      for g_i in range(400):
            communities = [random.randint(30, 81) for i in range(random.randint(2, 5))]
            list_adj.append(nx.adjacency_matrix(n_community(communities, p_inter=0.3, p_intera=0.05)))
            list_x.append(None)
            list_labels.append(len(communities)-2)

  elif graph_type == "PVGAErandomGraphs":
      for i in range(1000):
          import randomGraphGen
          # n = np.random.randint(low=20, high=40)
          n = 20
          graphGen = randomGraphGen.GraphGenerator()
          list_x.append(None)
          g, g_type = graphGen(n)
          list_adj.append(nx.adjacency_matrix(g))
          list_labels.append(g_type)
      # graphs_to_writeOnDisk = [gr.toarray() for gr in list_adj]
      # np.save('PVGAErandomGraphs.npy', graphs_to_writeOnDisk, allow_pickle=True)


    

  # elif graph_type == "PVGAErandomGraphs_10000":
  #     for i in range(10000):
  #         import randomGraphGen
  #         # n = np.random.randint(low=20, high=40)
  #         n = 20
  #         graphGen = randomGraphGen.GraphGenerator()
  #         list_x.append(None)
  #         list_adj.append(nx.adjacency_matrix(graphGen(n)))
  #     graphs_to_writeOnDisk = [gr.toarray() for gr in list_adj]
  #     np.save('PVGAErandomGraphs_10000.npy', graphs_to_writeOnDisk, allow_pickle=True)
  # elif graph_type == "PVGAErandomGraphs_100000":
  #     for i in range(100000):
  #         import randomGraphGen
  #         # n = np.random.randint(low=20, high=40)
  #         n = 20
  #         graphGen = randomGraphGen.GraphGenerator()
  #         list_x.append(None)
  #         list_adj.append(nx.adjacency_matrix(graphGen(n)))
  #     graphs_to_writeOnDisk = [gr.toarray() for gr in list_adj]
  #     np.save('PVGAErandomGraphs_100000.npy', graphs_to_writeOnDisk, allow_pickle=True)
  elif graph_type == 'small_lobster':
      graphs = []
      p1 = 0.7
      p2 = 0.7
      count = 0
      min_node = 8
      max_node = 12
      max_edge = 0
      mean_node = 15
      num_graphs = 8
      seed=1234
      seed_tmp = seed
      while count < num_graphs:
          G = nx.random_lobster(mean_node, p1, p2, seed=seed_tmp)
          if len(G.nodes()) >= min_node and len(G.nodes()) <= max_node:
              graphs.append(G)
              list_adj.append(nx.adjacency_matrix(G))
              list_x.append(None)
              count += 1
          seed_tmp += 1
  elif graph_type == 'small_lobster':
      graphs = []
      p1 = 0.7
      p2 = 0.7
      count = 0
      min_node = 1000
      max_node = 10000
      max_edge = 0
      mean_node = 5000
      num_graphs = 100
      seed=1234
      seed_tmp = seed
      while count < num_graphs:
          G = nx.random_lobster(mean_node, p1, p2, seed=seed_tmp)
          if len(G.nodes()) >= min_node and len(G.nodes()) <= max_node:
              graphs.append(G)
              list_adj.append(nx.adjacency_matrix(G))
              list_x.append(None)
              count += 1
          seed_tmp += 1
  elif graph_type == 'lobster':
      graphs = []
      p1 = 0.7
      p2 = 0.7
      count = 0
      min_node = 10
      max_node = 100
      max_edge = 0
      mean_node = 80
      num_graphs = 100
      seed=1234
      seed_tmp = seed
      while count < num_graphs:
          G = nx.random_lobster(mean_node, p1, p2, seed=seed_tmp)
          if len(G.nodes()) >= min_node and len(G.nodes()) <= max_node:
              graphs.append(G)
              list_adj.append(nx.adjacency_matrix(G))
              list_x.append(None)
              count += 1
          seed_tmp += 1
      # writing the generated graph for benchmarking
      # graphs_to_writeOnDisk = [gr.toarray() for  gr in list_adj]
      # np.save('Lobster_adj.npy', graphs_to_writeOnDisk, allow_pickle=True)
  elif graph_type=="mnist":
      list_adj = []
      list_x = []
      import torch_geometric
      dataset_b = torch_geometric.datasets.MNISTSuperpixels(root="data/geometric")
      for i in range(len(dataset_b.data.y)):  # len(dataset_b.data.y)
          in_1 = dataset_b[i].edge_index[0].detach().numpy()
          in_2 = dataset_b[i].edge_index[1].detach().numpy()
          valu = numpy.ones(len(in_2))
          adj = scipy.sparse.csr_matrix((valu, (in_1, in_2)), shape=(dataset_b[i].num_nodes, dataset_b[i].num_nodes))
          list_adj.append(adj)
          list_x.append(None)
  elif graph_type == "zinc":
      import torch_geometric
      dataset_b = torch_geometric.datasets.ZINC(root="data/geometric/MoleculeNet/zinc", subset=False)
      list_adj = []
      for i in range(len(dataset_b.data.y)):
          in_1 = dataset_b[i].edge_index[0].detach().numpy()
          in_2 = dataset_b[i].edge_index[1].detach().numpy()
          valu = numpy.ones(len(in_2))
          adj = scipy.sparse.csr_matrix((valu, (in_1, in_2)), shape=(dataset_b[i].num_nodes, dataset_b[i].num_nodes))
          list_adj.append(adj)
          list_x.append(None)
  elif graph_type == "cora":
      import input_data
      list_adj, list_x, _,_,_ = input_data.load_data(graph_type)
      list_adj = [list_adj]
      list_x = [list_x]
  elif graph_type == "ACM":
      import input_data
      list_adj, list_x, _,_,_ = input_data.load_data(graph_type)
      list_adj = [list_adj]
      list_x = [list_x]
  elif graph_type == 'ego':
      _, _, G = Graph_load(dataset='citeseer')
      # G = max(nx.connected_component_subgraphs(G), key=len)
      G = max((G.subgraph(c) for c in nx.connected_components(G)), key=len)
      G = nx.convert_node_labels_to_integers(G)
      graphs = []
      for i in range(G.number_of_nodes()):
          G_ego = nx.ego_graph(G, i, radius=3)
          if G_ego.number_of_nodes() >= 50 and (G_ego.number_of_nodes() <= 400):
              graphs.append(G_ego)
              list_adj.append(nx.adjacency_matrix(G_ego))
              list_x.append(None)


  elif graph_type == 'FIRSTMM_DB':
    list_adj, list_x, list_labels  = graph_load_batch(
        "data/Kernel_dataset/",
        min_num_nodes=0,
        max_num_nodes=2000,
        name='FIRSTMM_DB',
        node_attributes=False,
        graph_labels=True)

  elif graph_type == 'DD':
    list_adj, list_x, list_labels  = graph_load_batch(
        "data/Kernel_dataset/",
        min_num_nodes=100,
        max_num_nodes=500,
        name='DD',
        node_attributes=False,
        graph_labels=True)
    # args.max_prev_node = 230



  def return_subset(A,X,Y, limited_to):
      indx = list(range(len(A)))
      random.shuffle(indx)
      A = [A[i] for i in indx]
      X = [X[i] for i in indx]
      if Y!=None and len(Y)!=0 : Y = [Y[i] for i in indx]

      if limited_to != None:

          A = A[:limited_to]
          X = X[:limited_to]
          if Y!=None and len(Y)!=0 : Y = Y[:limited_to]
      return A,X,Y


  if return_labels ==True:
      if len(list_labels)==0:
          list_labels = None

  return return_subset(list_adj, list_x, list_labels, limited_to)

def data_split(graph_lis, list_x=None, list_label=None):
    #suffle the data
    random.seed(123)
    index = list(range(len(graph_lis)))
    random.shuffle(index)
    graph_lis = [graph_lis[i] for i in index]

    if list_x!=None:
        list_x = [list_x[i] for i in index]

    if list_label!=None:
        list_label = [list_label[i] for i in index]

    #----------------------------------------

    graph_test_len = len(graph_lis)

    graph_train = graph_lis[0:int(0.8 * graph_test_len)]  # train
    # graph_validate = graph_lis[0:int(0.2 * graph_test_len)]  # validate
    graph_test = graph_lis[int(0.8 * graph_test_len):]  # test on a hold out test set

    list_x_train = list_x_test = None
    if list_x!=None:
        list_x_train = list_x[0:int(0.8 * graph_test_len)]  # train
        list_x_test = list_x[int(0.8 * graph_test_len):]

    list_label_train = list_label_test = None
    if list_label!=None:
        list_label_train = list_label[0:int(0.8 * graph_test_len)]  # train
        list_label_test = list_label[int(0.8 * graph_test_len):]

    return  graph_train, graph_test, list_x_train , list_x_test,list_label_train, list_label_test

# list_adj, list_x = list_graph_loader("grid")
# list_graph = Datasets(list_adj,self_for_none, None)

def BFS(list_adj):
    if not list_adj:
        return list_adj

    # Check if the data is multi-view
    is_multi_view = isinstance(list_adj[0], list)

    for i in range(len(list_adj)):
        if is_multi_view:
            # For multi-view, compute BFS order on the first view (e.g., SC matrix)
            # and apply the same order to all views for that subject.
            first_view = list_adj[i][0]
            
            # Find a starting node that is not an isolate
            degrees = np.array(first_view.sum(axis=1)).flatten()
            potential_starters = np.where(degrees > 0)[0]
            if len(potential_starters) == 0:
                # If all nodes are isolated, just continue, no permutation needed
                continue
            start_node = random.choice(potential_starters)

            bfs_index = scipy.sparse.csgraph.breadth_first_order(first_view, start_node, return_predecessors=False)
            
            # Apply the same permutation to all views of this graph
            for v_idx in range(len(list_adj[i])):
                view_matrix = list_adj[i][v_idx]
                permuted_view = view_matrix[bfs_index, :][:, bfs_index]
                list_adj[i][v_idx] = permuted_view
        else:
            # Original logic for single-view graphs
            single_view_graph = list_adj[i]
            
            degrees = np.array(single_view_graph.sum(axis=1)).flatten()
            potential_starters = np.where(degrees > 0)[0]
            if len(potential_starters) == 0:
                continue
            start_node = random.choice(potential_starters)

            bfs_index = scipy.sparse.csgraph.breadth_first_order(single_view_graph, start_node, return_predecessors=False)
            list_adj[i] = single_view_graph[bfs_index, :][:, bfs_index]
            
    return list_adj

def BFSWithAug(list_adj,X_s, label_s, number_of_per = 1):
    list_adj_ = []
    X_s_ = []
    label_s_ = []
    for _ in range(number_of_per):
        for i, adj in enumerate(list_adj):
            mone_is_nodes = list(np.array(adj.sum(0)).reshape(-1))
            mone_is_nodes = [x for x in range(len(mone_is_nodes)) if mone_is_nodes[x] >= 1]
            node_i = random.choice(mone_is_nodes)
            bfs_index = scipy.sparse.csgraph.breadth_first_order(list_adj[i],node_i)
            list_adj_.append(list_adj[i][bfs_index[0],:][:,bfs_index[0]])


            X_s_.append(X_s[i])
            if label_s!=None:
                label_s_.append(label_s[i])
    if len(label_s_)==0:
        label_s_ = label_s
    return list_adj_, X_s_, label_s_

def permute(list_adj, X):
    for i, _ in enumerate(list_adj):
        p = list(range(list_adj[i].shape[0]))
        np.random.shuffle(p)

        list_adj[i] = list_adj[i][p, :]
        list_adj[i]= list_adj[i][:, p]
        # list_adj[i].eliminate_zeros()
        if X != None:
            X[i] = X[i][p, :]
            X[i] = X[i][:, p]
    return list_adj, X

def node_festure_creator(adj_in,steps=3, rand_dim=0, Use_identity = False, norm=None, uniform_size=False):

    if norm==None:
        norm=adj_in.shape[0]

    if not uniform_size:
        adj = adj_in
    else:
        adj = csr_matrix((norm, norm))
        adj[:adj_in.shape[0],:adj_in.shape[0]] +=adj_in

    traverse_matrix = adj
    featureVec=[np.array(adj.sum(1))/norm]
    for i in range(steps):
        traverse_matrix = traverse_matrix.dot(adj.transpose())
        feature = traverse_matrix.diagonal().reshape(-1,1)
        # converting it to one hot
        # one_hot = np.zeros((feature.size, int(feature.max()+1)))
        # one_hot[np.arange(one_hot.shape[0]),np.squeeze(np.asarray((feature).astype(int)))] = 1
        # one_hot.astype(int)
        featureVec.append(feature/norm**(i+1))
    if rand_dim>0:
        np.random.seed(0)
        featureVec.append(np.random.rand(adj.shape[-1], rand_dim))

    if Use_identity:
        featureVec.append(np.identity(norm))

    return numpy.concatenate(featureVec, 1)

def padd_adj_to(adj_list, size):
    uniformed_list = []
    for adj in adj_list:
        adj_padded = lil_matrix((size, size))
        adj_padded[:adj.shape[-1], :adj.shape[0]] = adj[:, :]
        adj_padded.setdiag(1)
        uniformed_list.append(adj_padded)
    return uniformed_list

def BFS_Permute( adj_s, x_s, target_kelrnel_val):
  for i in range(len(adj_s)):
      degree = np.array(adj_s[0].sum(0)).reshape(-1)
      connected_node = np.where(degree > 1)
      unconnected_nodes = np.where(degree == 1)

      bfs_index = scipy.sparse.csgraph.breadth_first_order(adj_s[i], random.choice(connected_node[0]))
      bfs_index = list(np.unique(bfs_index[0]) )+ list(unconnected_nodes[0])
      adj_s[i] = adj_s[i][bfs_index, :][:, bfs_index]
      x_s[i] = x_s[i][bfs_index, :]
      for j in range(len(target_kelrnel_val)-2):
          target_kelrnel_val[j][i] = target_kelrnel_val[j][i][bfs_index, :][:, bfs_index]


  return adj_s, x_s, target_kelrnel_val

def load_neuroimaging_data(data_path):#new
    """
    Loads multi-view neuroimaging data (SC, FC) and multi-label CSV.

    Args:
        data_path (str): The directory containing sc.pkl, fc.pkl, and labels.csv.

    Returns:
        tuple: A tuple containing:
            - list_adj (list): A list of multi-view graphs. Each element is [sc_matrix, fc_matrix].
            - list_x (list): A list of None, as there are no node features.
            - list_labels (list): A list of multi-label numpy arrays.
    """
    print("Loading Multi-view Neuroimaging Dataset...")

    # 1. Load the pickle files
    try:
        with open(os.path.join(data_path, 'sc_site_16.pkl'), 'rb') as f:
            sc_data = pickle.load(f)
        with open(os.path.join(data_path, 'fc_site_16.pkl'), 'rb') as f:
            fc_data = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not find SC or FC pickle files in '{data_path}'.")
        raise e

    # 2. Load the labels CSV
    try:
        labels_df = pd.read_csv(os.path.join(data_path, 'site_16_labels.csv'))
        # Set the subjectkey as the index for easy lookup
        labels_df.set_index('subjectkey', inplace=True)
    except FileNotFoundError as e:
        print(f"Error: Could not find labels.csv in '{data_path}'.")
        raise e

    # 3. Align subjects and create data lists
    list_adj = []
    list_labels = []
    
    # Get the intersection of subject keys to ensure we only use complete data
    common_subjects = sorted(list(set(sc_data.keys()) & set(fc_data.keys()) & set(labels_df.index)))
    
    if not common_subjects:
        raise ValueError("No common subjects found across SC, FC, and labels files.")
        
    print(f"Found {len(common_subjects)} common subjects.")

    for subject_key in common_subjects:
        sc_matrix = sc_data[subject_key]
        fc_matrix = fc_data[subject_key]

        # Convert matrices to sparse format (csr is efficient)
        # Assuming they are numpy arrays from the pickle file
        sc_sparse = sp.csr_matrix(sc_matrix)
        fc_sparse = sp.csr_matrix(fc_matrix)

        # Create the multi-view list for this subject
        multi_view_adj = [sc_sparse, fc_sparse]
        list_adj.append(multi_view_adj)
        
        # Get the multi-label vector
        subject_labels = labels_df.loc[subject_key].values.astype(np.float32)
        is_ill = 1.0 if np.sum(subject_labels) > 0 else 0.0
        list_labels.append(is_ill)

    # Node features are not present in this dataset
    list_x = [None] * len(list_adj)
    
    return list_adj, list_x, list_labels



if __name__ == '__main__':
    import numpy as np
    from itertools import combinations
    import plotter

    result = list_graph_loader("PVGAErandomGraphs")
    graph = np.load('C:\git\GRANon13\data/PVGAErandomGraphs.npy', allow_pickle=True)


    result = list_graph_loader("PVGAErandomGraphs_100000")

    for G in result[0]:
        G = nx.from_numpy_matrix(G.toarray())
        plotter.plotG(G,"DD")
    # ----------------------------------------
    import plotter
    result = list_graph_loader("triangular_grid")
    for G in result[0]:


        G = nx.from_numpy_matrix(G.toarray())
        plotter.plotG(G,"DD")
    #----------------------------------------
    result_ = list_graph_loader("QM9")
    result=list_graph_loader("NCI1")
    import plotter

    for i, G in enumerate(result[0]):
        G = nx.from_numpy_matrix(G.toarray())
        plotter.plotG(G, "test_graph", plot_it=True)

    result=list_graph_loader("triangular_grid")
    import plotter

    for i, G in enumerate(result[0]):
        G = nx.from_numpy_matrix(G.toarray())
        plotter.plotG(G, "test_graph")

    import torch_sparse
    import torch; print(torch.version.cuda)


    for i, graph in  enumerate(result[0]):
        print(nx.number_connected_components(nx.from_scipy_sparse_matrix(graph)))

    BFS(result[0])
    result = list_graph_loader("multi_community")
    Datasets(result[0], True, None,Max_num=None)
    Datasets.get__(0,2, True, None, None)
    for G in result[0]:


        G = nx.from_numpy_matrix(G.toarray())
        plotter.plotG(G,"DD")

