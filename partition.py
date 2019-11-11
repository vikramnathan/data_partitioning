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
                'be chosen to make the utilization about 0.5')
parser.add_argument('--max-block-size',
        type=int,
        default=-1,
        help='Max size per block. If set, must be larger than the largest region size')
parser.add_argument('--replicas',
        type=int,
        default=1,
        help='Number of replicas to optimize for')
parser.add_argument('--timeout-sec',
        type=int,
        default=300,
        help='Number of seconds to run the optimizer for')
args = parser.parse_args()


def insert_region_IDs(intersections):
    nregions = len(intersections)
    sizes = {}
    query_map = defaultdict(list)
    ix = 0
    for qs, size in intersections.items():
        sizes[ix] = size
        for q in qs:
            query_map[q].append(ix)
        ix += 1
    return query_map, sizes

def plot_cdf(intersections):
    xs = list(sorted(intersections.values()))
    ys = np.arange(0, len(xs))
    plt.semilogx(xs, ys, '.', markersize=2)
    plt.savefig('intersection_cdf.pdf')

def construct_ilp():
    #intscts, max_id = intersections.build_point_map_inmemory(args.query_dir)
    intscts, max_id = intersections.Inverter(args.query_dir,
            args.work_dir).run()
    query_map, sizes = insert_region_IDs(intscts)
    s = solver.PartitionSolver()
    s.set_region_sizes(sizes)
    s.set_query_regions(query_map)
    s.set_num_replicas(args.replicas)
    
    max_region_size = max(sizes.values())
    if args.max_block_size > 0 and args.max_block_size < max_region_size:
        print('Warning: --max-block-size set to %d, which is less ' + \
                'than the largest region size (%d). Adjusting...' % \
                (args.max_block_size, max_region_size))
    block_size = max(args.max_block_size, max_region_size)
    s.set_max_block_size(block_size)
    
    nblocks = args.num_blocks
    if nblocks < 0:
        nblocks = int(1 + sum(sizes.values()) * 2. / block_size) 
    s.set_num_blocks(nblocks)
    return s

def report(result, assignment=True):
    print('Objective =', result.objVal, ', MPI gap =', result.relative_gap_to_optimal)
    if assignment:
        for r, b in result.region_assignment.items():
            print('Region %d => Block %d' % (r, b))

s = construct_ilp()
r = s.solve(timeout_sec = args.timeout_sec)
report(r, assignment=False)
