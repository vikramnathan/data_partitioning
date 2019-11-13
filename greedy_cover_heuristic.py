import numpy as np
import sys
import os
import intersections
import itertools
from collections import defaultdict
from progress.bar import ShadyBar

## Implements a greedy heuristic to solve the region assignment problem.
# Given a graph where regions are nodes, and edges are weighted by how many
# queries two regions share, we greedily choose vertices to add to the cover
# until the cover reaches the maximum block size. Then we start a new cover.

class GreedyCoverHeuristic:
    # query regions is a map from query ID to list of region IDs. region_sizes
    # is a map from region ID to the number of points in that region.
    def __init__(self, query_regions, region_sizes, num_blocks, max_block_size):
        # Getting a key error here means that the regions are not properly IDed.
        self.vertex_sizes = [region_sizes[r] for r in range(len(region_sizes))]
        # Assume the graph is sparse, so represent edges as a list of neighbors
        # + weights. An edge (u, v) is indexed under both u and v.
        self.edges = [defaultdict(int) for _ in self.vertex_sizes]
        self.query_sets = [set() for _ in range(len(region_sizes))]
        self.populate_query_sets(query_regions)

        self.add_edges(query_regions)
        self.num_blocks = num_blocks
        self.num_vertices = len(region_sizes)
        self.block_size = max_block_size
        # Keeps track of all the regions that are assigned to the same block,
        # keyed by their current node ID
        self.merged = {k: [k] for k in range(len(region_sizes))}

    def populate_query_sets(self, query_regions):
        for q, rs in query_regions.items():
            for r in rs:
                self.query_sets[r].add(q)

    # Given two region IDs get their weight
    def get_edge_weight(self, u_id, v_id):
        return self.edges[u_id][v_id]

    def inc_edge_weight(self, u_id, v_id):
        self.edges[u_id][v_id] += 1 
        self.edges[v_id][u_id] += 1 

    def add_edges(self, query_regions):
        for q, rs in query_regions.items():
            for pair in itertools.combinations(rs, 2):
                self.inc_edge_weight(pair[0], pair[1])

    def max_weight_edge(self):
        max_w = -1
        argmax_e = None
        for u, es in enumerate(self.edges):
            if es is None:
                continue
            for v, w in self.edges[u].items():
                if w > max_w:
                    max_w = w
                    argmax_e = (u, v)
        return argmax_e, max_w

    # Returns a list of neighbors, in decreasing order of weight.
    def weighted_neighbors(self, v):
        neighbors = [(u, w) for u, w in self.edges[v].items()]
        return sorted(neighbors, reverse=True, key=lambda p: p[1])

    # Given an edge (u,v), merge u and v into a single node, with a new ID
    # min(u,v). The weight of the new edge is the max of the weight of the
    # previous two edges.
    def merge_nodes(self, u, v):
        new_id = min(u, v)
        old_id = max(u, v)
        self.query_sets[new_id] = self.query_sets[new_id].union(
                self.query_sets[old_id])
        self.query_sets[old_id] = None
        for e, _ in self.edges[old_id].items():
            if e != new_id:
                new_w = len(self.query_sets[new_id].intersection(
                    self.query_sets[e]))
                self.edges[e][new_id] = new_w
                self.edges[new_id][e] = new_w
            del self.edges[e][old_id]
        self.edges[old_id] = None
        self.vertex_sizes[new_id] += self.vertex_sizes[old_id]
        self.vertex_sizes[old_id] = None
  
        self.merged[new_id].extend(self.merged[old_id])
        del self.merged[old_id]
        self.num_vertices -= 1
        return new_id

    def remove_node(self, v):
        for e in self.edges[v].keys():
            del self.edges[e][v]
        self.edges[v] = None
        self.num_vertices -= 1

    def get_assignment(self):
        asst = {}
        block_ix = 0
        keys = list(sorted(self.merged.keys(),
                reverse=True,
                key=lambda v: self.vertex_sizes[v]))
        done = set()
        for i, k in enumerate(keys):
            if k in done:
                continue
            blck_size = self.vertex_sizes[k]
            for r in self.merged[k]:
                assert r not in asst
                asst[r] = block_ix
            # Packing heuristic to pack some of the small groups in the same
            # block, preferring larger ones.
            for i_back in range(i+1, len(keys)):
                if i_back in done:
                    continue
                grp = keys[i_back]
                if blck_size + self.vertex_sizes[grp] <= self.block_size:
                    for r in self.merged[grp]:
                        assert r not in asst
                        asst[r] = block_ix
                    print('Adding', self.merged[grp], 'to', self.merged[k])
                    done.add(grp)
                    blck_size += self.vertex_sizes[grp]
            block_ix += 1
        print('Used %d of %d blocks' % (block_ix, self.num_blocks))
        assert block_ix <= self.num_blocks,\
                "Block assignment took %d blocks, did not fit " % block_ix + \
                "within allocation of %d" % self.num_blocks
        return asst

    def check_valid(self):
        # Sanity check to make sure the graph structure is consistent.
        valid_nodes = set(i for i in range(len(self.edges)) if
                self.edges[i] is not None)
        for n, es in enumerate(self.edges):
            if es is None:
                continue
            for v, w in es.items():
                assert v in valid_nodes
                assert self.edges[v][n] == w
        for u, vs in self.merged.items():
            assert u in valid_nodes
            for v in vs:
                # v all nodes referenced in the merged node should have been
                # removed.
                if v != u:
                    assert v not in valid_nodes

    
    def solve(self):
        cur_v = None
        bar = ShadyBar('Running Greedy Cover heuristic', max=self.num_vertices,
                suffix='%(percent)d%%')
        # Use the max weight edge as a starting point
        while self.num_vertices > 0:
            if cur_v is None:
                e, w = self.max_weight_edge()
                if e is None:
                    print('Num Vertices remaining: %d' % self.num_vertices)
                    cur_v = [r for r in range(len(self.edges)) if \
                            self.edges[r] is not None][0]
                else:
                    cur_v = e[0]
            nbrs = self.weighted_neighbors(cur_v)
            # Visit neighbors in decreasing order until we find one
            # that fits within a block.
            found = False
            for n, w in nbrs:
                assert self.edges[n] is not None
                if self.vertex_sizes[cur_v] + self.vertex_sizes[n] <= \
                        self.block_size:
                    # Merge cur_v and n
                    cur_v = self.merge_nodes(cur_v, n)
                    found = True
                    break
            if not found:
                # Done filling this block. Delete this node and move on.
                self.remove_node(cur_v)
                cur_v = None
            #self.check_valid()
            bar.next()
        bar.finish()

        return self.get_assignment()
        

