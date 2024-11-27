import numpy as np 
from scipy.spatial.distance import cdist
import igraph as ig
import tqdm

class GraphMaxShift:
    """
    Partition graph according to the Graph Max Shift method.
    
    The implementation is for the general case of a weighted graph.
    
    From each node i, start at i_0 = i and then at iteration t,
    move to a node in the r-neighborhood with the highest degree (as determined by the h-neighborhood). 
    Typically, we set h = r. 
    
    At the end, we merge together any modes which are within m hops of one another.
    
    In this implementation, two tie-breaking mechanisms are supported.
    In the first (simpler) method, to break a tie between nodes i and j, we choose node i if i < j. 
    In the second method, all paths are followed. Thus, a node may belong to multiple clusters.
    
    STILL TO DO: Update the m-hop merging in the second tie breaking method. 
    """
    
    
    def __init__(self,graph, tie_method = 0):
        
        """
        Inputs
        graph: object of Geometric Graph class
        tie_method: 0 (first method) or 1 (second method)
        """
        
        self.graph = graph 
        self.n = graph.n
        self.tie_method = tie_method
        
    def update_graph(self,r, h=None):
        self.r = r 
        if h == None:
            self.h = r
        else:
            self.h = h 
        self.graph.update_graph(r, h)
        
    def _init_clusters(self):
        if self.tie_method == 1:
            self.clusters = {i:[] for i in np.arange(self.n)}
        else:
            self.clusters = np.ones(self.n)*-1 
            
    def cluster(self,r, h = None, m = 1):
  
        """
        Cluster the data with Graph Max Shift parameters h,r, and m.
        
        Cluster labels are arbitrary and vary with the tie method.
        """
        
        self.update_graph(r, h) #update degree and neighbors of the graph
        self._init_clusters()
        
        if self.tie_method == 1:

            self._get_modes() #identfy modes
            self._get_mode_labels(m) #group together labels for modes that are within a distance r

            for i in tqdm.tqdm(range(self.n)):
                self.clusters[i] = self._get_cluster_all_paths(i)
            
        else:
            for i in tqdm.tqdm(range(self.n)):
                self.clusters[i] = self._get_cluster_simple(i)
  
            # store mode information 
            self.is_mode = self.clusters == np.arange(self.n) 
            self.is_mode = self.is_mode.astype(bool)
            self.modes = np.where(self.is_mode)[0]
            
            # merge together modes that are within m hops of one another 
            if m >0:
                self._merge_modes(m)
        
    
    
    
    ###### Tie breaking method 1: break ties by going to node with smallest node label ######
    
    def _one_step_simple(self, curr_idx):
        """
        Take one step of Graph Max Shift with the first tie-breaking method
        Returns next node in seqence (int), and if the starting index is_mode (boolean, if have hit end of path or not)
        """

        curr_deg = self.graph.deg[curr_idx]
        nbhd = np.array(self.graph.nbhds[curr_idx]) 
        nbhd_deg = self.graph.deg[nbhd]
        argmax = nbhd[np.where(nbhd_deg== nbhd_deg.max())[0]]
        idx = np.min(argmax) # break ties with minimum node index

        if curr_idx == idx:
            return idx, True 
        else:
            return idx, False
        
        
    def _get_cluster_simple(self, curr_idx ): # do we need prev explored with this updated version of the code??
        """
        Cluster a node with the first tie breaking method 
        """
        
        if self.clusters[curr_idx] != -1: # if reach a point have already visited, return cluster label
            return self.clusters[curr_idx]

        curr_idx, is_mode = self._one_step_simple(curr_idx)
        while not is_mode:
            curr_idx, is_mode = self._one_step_simple(curr_idx)
        return curr_idx
    
    def _merge_modes(self,n_hops = 1):
        temp_labels = {m:i for i,m in enumerate(self.modes) }
        inv_temp_labels = {v: k for k, v in temp_labels.items()}
        
        edges = []

        for idx, i in enumerate(self.modes):
            for j in self.modes[idx+1:]:
                graph_distance = len(self.graph.G.get_shortest_path(i,j)) -1
                if (graph_distance <= n_hops) & (graph_distance > 0) :
                    edges += [(temp_labels[i],temp_labels[j])]


        mode_graph = ig.Graph(n = len(self.modes), edges = edges)
        cc = list(mode_graph.connected_components())
        
        for c in cc:
            if len(c)>1:
                mode_labels = [inv_temp_labels[i] for i in c]
                self.clusters[np.isin(self.clusters, mode_labels)] = mode_labels[0]
    
    ###### Tie breaking method 2: follow all paths ######
        
    def _get_cluster_all_paths(self, curr_idx, prev_explored = None):
        """
        Cluster a node with the second tie breaking method 
        """
        
        if len(self.clusters[curr_idx]) !=0:  # if reach a point have already visited, return its cluster label
            return self.clusters[curr_idx]
        
        if prev_explored == None:
            prev_explored = []
        clusters = []
        
        curr_deg = self.graph.deg[curr_idx]
        nbhd = np.array(list(self.graph.nbhds[curr_idx]))
        nbhd_deg = self.graph.deg[nbhd]
        argmax = nbhd[np.where(nbhd_deg== nbhd_deg.max())[0]]

        if curr_deg ==  nbhd_deg.max():
            clusters += [self.mode_labels[curr_idx]] #must be at a mode in this case

            if np.sum(self.is_mode[argmax]) < len(argmax): # if all neighbors are also modes

                mode_idxs = [idx for idx in argmax if self.is_mode[idx]]
                non_mode_idxs = [idx for idx in argmax if idx not in mode_idxs]
                prev_explored += mode_idxs
                for idx in non_mode_idxs:
                    if idx not in prev_explored:
                        clusters += self._get_cluster_all_paths(idx, prev_explored)
                        prev_explored += [idx]          

        else:
             for idx in argmax:
                clusters += self._get_cluster_all_paths(idx, prev_explored)
                prev_explored += [idx]
     
        return list(np.unique(clusters))
        
 
    def _get_modes(self):        
        self.is_mode = np.zeros((self.n))
        for i in range(self.n):
            deg = self.graph.deg[i]
            nbhd = self.graph.nbhds[i]
            nbhd_deg = self.graph.deg[nbhd]
            if deg ==  nbhd_deg.max():
                self.is_mode[i] = 1
        self.is_mode = self.is_mode.astype(bool)
        self.modes = np.where(self.is_mode)[0] #get indices of modes
      
    def _get_mode_labels(self, n_hops):
        """
        For each identified mode, this function labels the modes so that if two modes are within n_hops, they have the same label. 
        """
        
        self.mode_labels = np.ones(self.n)*-1
        temp_labels = {m:i for i,m in enumerate(self.modes) }
        inv_temp_labels = {v: k for k, v in temp_labels.items()}

        
        #construct a graph on the modes and identify connected components 
        edges = []
        
        for idx, i in enumerate(self.modes):
            for j in self.modes[idx+1:]:
                graph_distance = len(self.graph.G.get_shortest_path(i,j)) -1
                if (graph_distance <= n_hops) & (graph_distance > 0) :
                    edges += [(temp_labels[i],temp_labels[j])]

        mode_graph = ig.Graph(n = len(self.modes), edges = edges)
        cc = list(mode_graph.connected_components())

        # connected components are neighbors and modes should have the same label 
        for i, component in enumerate(cc):
            for node in component:
                self.mode_labels[inv_temp_labels[node]] = i 
    


    ###### Visualization tools ######
    def reindex_clusters(self, min_count):
        """
        To be used to aid in visualizations. 
        Reindexes clusters labels so that clusters with less than min_count points all have the label zero.
        Nodes assigned to multiple clsuters have label -1.
        All other clusters are re-indexed to have labels 1 through num_clusters.
        """
        
        if self.tie_method == 1:
            clusters = []
            tie_count = 0 
            for i in range(self.n):
                if len(self.clusters[i]) > 1:
                    clusters += [-1]
                    tie_count += 1
                else:
                    clusters += [self.clusters[i][0]]
            
            
            vals, counts = np.unique(clusters, return_counts= True) 
            
            accepted_vals = vals[counts>min_count]
            accepted_counts = counts[counts>min_count]
            
            #keep points with ties
            if -1 not in accepted_vals:
                accepted_vals = np.append(accepted_vals, -1)
                accepted_counts =  np.append(accepted_counts, tie_count)

        else: 
            clusters = self.clusters
            vals, counts = np.unique(clusters, return_counts= True)

            accepted_vals = vals[counts>min_count]
            accepted_counts = counts[counts>min_count]
           
            
        reindexed_clusters = []

        # label according to counts 
        labels = {x:i+1 for i,x in enumerate(accepted_vals[np.argsort(accepted_counts)[::-1]])}
        
        if self.tie_method == 1:
            labels[-1] = -1 # keep -1 label
        for i in clusters:
            if i not in accepted_vals:
                reindexed_clusters += [0]
            else:
                reindexed_clusters  += [labels[i]]
        reindexed_clusters = np.array(reindexed_clusters)
        return reindexed_clusters
    
    
    
    def get_path(self, curr_idx):
        """
        Starting from curr_idx, get the path from a starting node to the mode it is clustered with
        """
        
        if self.tie_method == 1:
            return self._path_all(curr_idx)
        else:
            return self._path_simple(curr_idx)

    def _path_simple(self, curr_idx):
        path = [curr_idx]
        _, is_mode = self._one_step_simple(curr_idx)
        while not is_mode:
            curr_idx, is_mode = self._one_step_simple(curr_idx)
            if not is_mode:
                path += [curr_idx]
        return path 
    
    def _path_all(self, curr_idx, prev_explored = None):
        if prev_explored == None:
            prev_explored = []
            
        paths = []
        path = [curr_idx]
        
        curr_deg = self.graph.deg[curr_idx]
        nbhd = np.array(list(self.graph.nbhds[curr_idx]))
        nbhd_deg = self.graph.deg[nbhd]
        argmax = nbhd[np.where(nbhd_deg== nbhd_deg.max())[0]]
        if curr_deg ==  nbhd_deg.max():
            paths += [path]
            if np.sum(self.is_mode[argmax]) < len(argmax): # if all neighbors are also modes

                mode_idxs = [idx for idx in argmax if self.is_mode[idx]]
                non_mode_idxs = [idx for idx in argmax if idx not in mode_idxs]
                prev_explored += mode_idxs
                for idx in non_mode_idxs:
                    if idx not in prev_explored:
                        temp_paths = self._path_all(idx, prev_explored)
                        prev_explored += [idx]
                        paths += [path + x for x in temp_paths]
        else:
             for idx in argmax:
                temp_paths = self._path_all(idx, prev_explored)
                prev_explored += [idx]
                paths += [path + x for x in temp_paths]
        return paths

class GeometricGraph:
    """
    Constructs a graph with nodes corresponding to the given data point.
    The degree of node i is the number of data points within a distance h from data point i. 
    The neighbors of node i are the nodes that correspond to the data points within a distance r from data point i. 
    Typically we set h = r. 
    
    We store distances so that h and r may be changed and the graph can be updated quickly. 
    Due to memory constraints, we only store distances less than a specified max_dist. 
    """
    
    
    def __init__(self, data, max_dist, batch_size = 10000):
        self.n = len(data)
        self.max_dist = max_dist
        self.h = None # use to determine degree, should be <= max_dist
        self.r = None  # use to determine neighbors, should be <= max_dist 
        self.data = data 
        self._init_graph(batch_size, max_dist)
        
    def _init_graph(self,  batch_size, max_dist):
        """
        Store edge (i,j) if || data[i] -data[j] || < max_dist.
        
        Batch_size should be adjusted according to memory constraints 
        """
        
        self.sources = []
        self.targets = []
        self.dists = []
        num_batches = self.n // batch_size + 1

        for i in tqdm.tqdm(range(num_batches)):
            start = i * batch_size
            end = min((i + 1) * batch_size, self.n)
            batch_data = self.data[start:end]
            batch_distances = cdist(batch_data, self.data)  # Compute pairwise distances between the current batch and all points
            mask = batch_distances < max_dist

            batch_sources, batch_targets = np.nonzero(mask) # get edges
            batch_flag = (batch_sources + start) < batch_targets  # by convention, store source < target 

            batch_sources, batch_targets =  batch_sources[batch_flag], batch_targets[batch_flag] 
            self.dists += batch_distances[batch_sources, batch_targets].tolist()

            batch_sources += start  #reindex according to the batch  
            self.sources += batch_sources.tolist()
            self.targets += batch_targets.tolist()


        self.sources = np.array(self.sources)
        self.targets = np.array(self.targets)
        self.dists = np.array(self.dists)
        
    def update_graph(self, r, h = None):
        
        self.r = r
        if h == None:
            self.h = r
        else:
            self.h = h

        edgelist = zip(self.sources[self.dists<self.r], self.targets[self.dists<r])
        self.G = ig.Graph(n = self.n, edges = edgelist)
        self._update_deg()
        self._update_nbhds()


    def _update_deg(self):
        if self.h == self.r:
            self.deg = np.array([self.G.degree(i) for i in np.arange(self.n)])
        else:
            edgelist = zip(self.sources[self.dists<self.h], self.targets[self.dists<self.h])
            G_h = ig.Graph(n = self.n, edges = edgelist)
            self.deg = np.array([G_h.degree(i) for i in np.arange(self.n)])
            

    def _update_nbhds(self):
        self.nbhds = {i:[x for x in self.G.neighbors(i)] + [i] for i in range(self.n)} # for each of computations, store self as a neighbor as well 

       