from collections import defaultdict
import numpy as np
import argparse
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import solver
import intersections

parser = argparse.ArgumentParser('Compute Intersections')
parser.add_argument('--query-dir',
        type=str,
        required=True,
        help='Directories with one file per query with the list of record IDs matching that query (binary)')
parser.add_argument('--work-dir',
        type=str,
        default='.cmp_intr_work_dir',
        help='Scratch directory into which intermediate results will be stored')
parser.add_argument('--num-blocks',
        type=int,
        default=-1,
        help='Number of blocks to partition data into. If not given, will ' + \
                'be chosen so that the avg utilization is --utilization')
parser.add_argument('--max-block-size',
        type=int,
        default=-1,
        help='Max size per block. If not set, defaults to the number of ' + \
                'points in the largest region')
parser.add_argument('--utilization',
        type=float,
        default=0.5,
        help='Target average number of points per block. This is used only ' + \
                'to determine the number of blocks if --num-blocks is not set')
parser.add_argument('--replicas',
        type=int,
        default=1,
        help='Number of replicas to optimize for')
parser.add_argument('--timeout-sec',
        type=int,
        default=300,
        help='Number of seconds to run the optimizer for')
args = parser.parse_args()

# Given a mapping of (tuples of query IDs) -> number of points in that
# intersection, split it into two maps by giving each region an ID:
# - a map from query ID to the region IDs 
# - a map from region ID to the number of points in that region.
# It may be the case that the size of a region is larger than the max allowed
# bock size (unlimited if set to the default). In this case, a single
# region may be split into multiple regions with size max_size, plus another
# region with any remaining points.
def insert_region_IDs(intersections, max_size=-1):
    if max_size < -1:
        # By default, there is no maximum region size.
        max_size = sys.maxint
    nregions = len(intersections)
    sizes = {}
    query_map = defaultdict(list)
    ix = 0
    # Some regions are larger than the maxmimum block size. In this case,
    # greedily put as much as possible into a single block, and just send the
    # remainder to the solver.
    preassigned_blocks = defaultdict(int)
    for qs, size in intersections.items():
        pts_left = size
        while pts_left > max_size:
            preassigned_blocks[qs] += 1
            pts_left -= max_size
        if pts_left > 0:
            sizes[ix] = pts_left
            for q in qs:
                query_map[q].append(ix)
            ix += 1

    return query_map, sizes, preassigned_blocks

def plot_cdf(intersections):
    xs = list(sorted(intersections.values()))
    ys = np.arange(0, len(xs))
    plt.semilogx(xs, ys, '.', markersize=2)
    plt.savefig('intersection_cdf.pdf')

def construct_ilp():
    #intscts, max_id = intersections.build_point_map_inmemory(args.query_dir)
    intscts, max_id = intersections.Inverter(args.query_dir,
            args.work_dir).run()
    query_map, sizes, preassigned = insert_region_IDs(intscts, max_size=args.max_block_size)
    s = solver.PartitionSolver()
    s.set_region_sizes(sizes)
    s.set_query_regions(query_map)
    s.set_num_replicas(args.replicas)
    
    max_region_size = max(sizes.values())
    block_size = max(args.max_block_size, max_region_size)
    s.set_max_block_size(block_size)
    
    nblocks = args.num_blocks
    # If not set, the number of blocks is determined by taking the regions that
    # need to still be assigned, and aiming for the given utilization.
    if nblocks < 0:
        nblocks = int(1 + sum(sizes.values()) / args.utilization / block_size) 
    s.set_num_blocks(nblocks)
    print('%d blocks preassigned' % sum(preassigned.values()))
    return s, preassigned

def report(result, preassignment, print_assignment=True):
    print('Solver Objective =', result.objVal, ', MPI gap =', result.relative_gap_to_optimal)
    if assignment:
        for r, b in result.region_assignment.items():
            print('Region %d => Block %d' % (r, b))
    num_blocks_accessed = int(result.objVal)
    # We have to add the blocks that are preassigned.
    for qs, b in preassignment.items():
        num_blocks_accessed += len(qs) * b
    print('Total # blocks accessed:', num_blocks_accessed)

s, preassigned = construct_ilp()
r = s.solve(timeout_sec = args.timeout_sec)
report(r, preassigned, print_assignment=False)
