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
import greedy_cover_heuristic as gch
import rowkey_partition_baseline as rpb

ALGS = [ "ilp", "greedy", "rowkey" ]
# Folder under which all work dirs live
WORK_DIR_BASE = 'scratch'

parser = argparse.ArgumentParser('Compute Intersections')
parser.add_argument('--query-dir',
        type=str,
        required=True,
        help='Directories with one file per query with the list of record IDs matching that query (binary)')
parser.add_argument('--alg',
        required=True,
        choices=ALGS,
        help="Algorithm to run")
parser.add_argument('--work-dir',
        type=str,
        default='tmp_work_dir',
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
parser.add_argument('--output-assignment',
        default='',
        type=str,
        help="File to print the region -> block assignments")

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

def construct_ilp(query_map, sizes, max_block_size, nblocks):
    #intscts, max_id = intersections.build_point_map_inmemory(args.query_dir)
    return s, preassigned

def report_cost(query_regions, assignment, preassignment):
    num_blocks_accessed = 0
    for rs in query_regions.values():
        blocks = set()
        for r in rs:
            blocks.add(assignment[r])
        num_blocks_accessed += len(blocks)
    # We have to add the blocks that are preassigned.
    for qs, b in preassignment.items():
        num_blocks_accessed += len(qs) * b
    print('Total # blocks accessed:', num_blocks_accessed)

def write_assignment(assignment, preassignment, region_sizes):
    with open(args.output_assignment) as out:
        out.write('# Format: Region ID, Region Size, Block ID\n')
        for r, b in assignment.items():
            out.write('%d, %d, %d\n' % (r, region_sizes[r], b)) 

def run_ilp():
    work_dir = os.path.join(WORK_DIR_BASE, args.work_dir)
    intscts, max_id = intersections.Inverter(args.query_dir,
            work_dir).run()
    query_map, sizes, preassigned = insert_region_IDs(intscts, max_size=args.max_block_size)
    print('%d blocks preassigned' % sum(preassigned.values()))
    
    max_region_size = max(sizes.values())
    block_size = max(args.max_block_size, max_region_size)
    
    nblocks = args.num_blocks
    # If not set, the number of blocks is determined by taking the regions that
    # need to still be assigned, and aiming for the given utilization.
    if nblocks < 0:
        nblocks = int(1 + max_id / block_size / args.utilization)
    # Can't use blocks that were already preassigned.
    nblocks -= sum(preassigned.values())
    
    s = solver.PartitionSolver()
    s.set_max_block_size(block_size)
    s.set_region_sizes(sizes)
    s.set_query_regions(query_map)
    s.set_num_replicas(args.replicas)
    s.set_num_blocks(nblocks)
    result = s.solve(timeout_sec = args.timeout_sec)
    print('Solver Objective =', result.objVal, ', MPI gap =', result.relative_gap_to_optimal)
    if len(args.output_assignment) > 0:
        write_assignment(result.region_assignment, preassigned, sizes)
    report_cost(query_map, result.region_assignment, preassigned)

def run_greedy_cover():
    work_dir = os.path.join(WORK_DIR_BASE, args.work_dir)
    intscts, max_id = intersections.Inverter(args.query_dir,
            work_dir).run()
    query_map, sizes, preassigned = insert_region_IDs(intscts, max_size=args.max_block_size)
    if args.max_block_size < 0:
        print('Must specify max_block_size')
        sys.exit(1)
    nblocks = args.num_blocks
    if nblocks < 0:
        nblocks = int(1 + max_id / args.max_block_size / args.utilization)
    # Can't use the blocks that were already preassigned.
    nblocks -= sum(preassigned.values())

    g = gch.GreedyCoverHeuristic(query_map, sizes, nblocks, args.max_block_size)
    asst = g.solve()
    report_cost(query_map, asst, preassigned)

if args.alg == "ilp":
    run_ilp()
elif args.alg == "greedy":
    run_greedy_cover
elif args.alg == "rowkey":
    assert args.max_block_size > 0, "--max-block-size must be set " + \
            "for rowkey partition baseline"
    rpb.compute_cost(args.query_dir, args.max_block_size)

