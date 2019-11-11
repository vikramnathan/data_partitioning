import numpy as np
import gurobipy as grb

class Result:
    def __init__(self, obj, asgn, gap_rel):
        # Objective value
        self.objVal = obj
        # Map from region ID to block ID 
        self.region_assignment = asgn
        self.relative_gap_to_optimal = gap_rel

class PartitionSolver:
    def __init__(self):
        # Map from region ID to number of points in that region
        self.region_sizes = {}
        # For each query, a list / set of the region IDs it includes.
        self.query_regions = {}
        self.num_replicas = None
        self.num_blocks = None

    # Map from region id to the size of that region.
    def set_region_sizes(self, region_sizes):
        assert (len(region_sizes) == max(region_sizes.keys())+1)
        assert (all(s > 0 for s in region_sizes.values()))
        self.region_sizes = region_sizes

    # Map from query id to the regions it contains
    def set_query_regions(self, regions):
        assert (len(regions) == max(regions.keys())+1), \
                "Got %d regions with max key %d" % (len(regions),
                        max(regions.keys()))
        assert all(r >= 0 for r in regions.keys()), \
                "Got negative query id"
        assert all(isinstance(v, list) or isinstance(v, set) for v in
                regions.values())
        self.query_regions = regions

    def set_num_blocks(self, b):
        self.num_blocks = b

    def set_num_replicas(self, repl):
        self.num_replicas = repl

    def set_max_block_size(self, pts):
        # Sets the maximum number of points per block
        self.max_block_size = pts

    def check_setup(self):
        for q, rs in self.query_regions.items():
            for r in rs:
                if r not in self.region_sizes:
                    print('Query %d references undefined region %d', q, r)
        print('Region check passed')

    def build_model(self):
        m = grb.Model('data_partition')
        print('Got %d regions, %d queries, %d blocks, max block size %d' % \
                (len(self.region_sizes), len(self.query_regions),
                    self.num_blocks, self.max_block_size))
        # Add a binary variable i, j if query i intersects block j
        Q = {}
        R = {}
        r = 0
#        for r in range(self.num_replicas):
        for i in range(len(self.query_regions)):
            for j in range(self.num_blocks):
                Q[r, i, j] = m.addVar(
                        vtype=grb.GRB.BINARY,
                        name='q_%d_%d_%d' % (r, i, j))

        # Add a binary variable k, j if region k is included in block j
        for k in range(len(self.region_sizes)):
            for j in range(self.num_blocks):
                R[r, k, j] = m.addVar(
                        vtype=grb.GRB.BINARY,
                        name='r_%d_%d_%d' % (r, k, j))

        # Add the constraint: query i touches block j if any regions it includes
        # are contained in block j.
        for i in range(len(self.query_regions)):
            for j in range(self.num_blocks):
                m.addConstr(
                        grb.quicksum(R[r,k,j] for k in self.query_regions[i]) \
                        <= len(self.query_regions[i]) * Q[r,i,j])

        # Add the constraint that the number of points in each block is less
        # than the limit.
        for j in range(self.num_blocks):
            m.addConstr(
                    grb.quicksum(n * R[r, k, j] for k, n in \
                        self.region_sizes.items()) <= self.max_block_size)

        # Add the constraint that every region must belong to exactly one block.
        for k in range(len(self.region_sizes)):
            m.addConstr(
                    grb.quicksum(R[r, k, j] for j in range(self.num_blocks)) == 1)
   

        # For each query, add a variable that's the minimum of the touched
        # blocks for the various replicas
#        X = {}
#        for i in range(len(self.query_regions)):
#            X[i] = m.addVar(vtype=grb.GRB.INTEGER)
#            m.addConstr(X[i] == \
#                grb.min_([
#                    grb.quicksum(Q[r,i,j] for j in range(self.num_blocks))
#                for r in range(self.num_replicas)]))
#        
#        m.setObjective(grb.quicksum(X[i] for i in \
#            range(len(self.query_regions)), grb.GRB.MINIMIZE)
        m.setObjective(grb.quicksum(Q[0, i, j] for i in \
            range(len(self.query_regions)) for j in range(self.num_blocks)))

        return m

    def solve(self, timeout_sec=1000):
        m = self.build_model() 
        m.setParam('TimeLimit', timeout_sec)
        m.optimize()
        
        # The solution is a mapping of region ID to block ID
        solution = {}
        for k in range(len(self.region_sizes)):
            found = False
            for j in range(self.num_blocks):
                v = m.getVarByName('r_%d_%d_%d' % (0, k, j))
                if v.x > 0:
                    # Make sure this region isn't assigned to multiple blocks
                    if k in solution:
                        print('WARNING: Multiple block assignments ' + \
                                'for region %d' % k)
                    solution[k] = j
                    found = True
            assert (found)

        return Result(m.objVal, solution, m.MIPGap)


