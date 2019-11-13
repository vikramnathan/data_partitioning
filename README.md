# Data partitioning

## Requirements
You just need Python 3.x and the following modules:
```
pip3 install numpy progress
```

## Sampling

If the dataset is too big, you might want to sample the query workloads prior to
running them. To generate sampled queries:
```
python dedup.py --source-dir=<original query dir> --output-dir=<output
query_dir> --sample=<sampling ratio>
```
Make sure the output query dir doesn't exist first.

## Point -> region reduction

The original ILP assigned a block to each individual record. However, this is
intractable for large numbers of points. Our workaround is to instead consider
assigning _regions_ to blocks, where a _region_ consists of all the points that
are touched by the same set of queries. For example, if Q1 touches {1, 2, 3} and
Q2 touches {2, 3, 4}, there are 3 total regions: {1} touched by (Q1), {2, 3} touched by (Q1, Q2), and {4} touched by (Q2). Note that we do not consider points not touched by any query.

Using regions has some key advantages over the point-based ILP formulation:
- There are typically much fewer regions than points, making the ILP tractable
  even on a large dataset.
- A mapping from region to block yields a function that can assign blocks to new
  points.
- The formulations are roughly equivalent. By definition, any queries that touch
  a region must fetch all the points in that region. Each region is effectively
  an equivalence class of points and should be assigned to a single block. This
  is only a _rough_ equivalence because of the edge cases below.

The above reduction differs slightly from the point based formulation in a
couple ways:
- It is typically impossible to get perfect 100% utilization of each block
  with this formulation. This is also true for other high utilization
  requirements, since each region is assigned to only one block and cannot be
  split. 
- Hard bounds on minimum block utilization can
  yield an infeasible ILP, even though the point-based ILP problem would have had
  a solution. There are two workarounds:
    * Enforce an _average_ utilization instead of a minimum utilization per
      block. Average utilizations can be enforced by picking a max block size
      and choosing the number of blocks that gives the target utilization. This
      is our current approach.
    * Break the regions up into smaller regions of size at most
      `max_block_size * (1 - min_utilization)`. This ensures that the resulting
      ILP is always feasible.


## Running

```
python partition.py --query-dir --alg={rowkey,greedy,ilp} [ --work-dir ]
    [ --timeout-sec ] [ --max-block-size ] [ --num-blocks]
```
Arguments:
- `--query-dir` is a directory containing only the query files: one file for each
query with the row IDs of the records returned by that query. Each file should
be in binary format, with each ID being an int32. It is unpacked with
`np.fromfile(.., dtype=np.int32)`
- `--alg` is the algorithm to use:
    - 'rowkey' is the baseline scheme where each contiguous segment of
      `--max-block-size` records are allocated to the same block. Note that
      `max-block-size` must be set for this option.
    - 'greedy' runs the greedy graph cover heuristic. The graph consists of
      regions as vertices with edges between regions that share a query,
      weighted by the number of queries they share. Starting from a cover with
      a single region, the heuristic adds the neighbor that shares the most
      queries with the running cover, and expands the cover to include it, stopping
      and starting a new cover when the max block size is reached.
    - 'ilp' runs the ILP solver. 
- `--work-dir` is a directory with intermediate results. All work dirs are
  made subdirectories of the `scratch` directory since it is ignored by git. The work
  dir is also used to cache some results so we don't have to invert the queries
  every time. Use the same work dir on the same query dir to avoid needless recomputation.
- `--timeout-sec` is the timeout passed to the Gurobi optimizer. After this
  amount of time, the best solution found so far is returned. Note that this
  timeout applies to _only_ the Gurobi ILP solver and not other components, like
  query inversion or ILP setup. This option only has an effect if `--alg=ilp`.
- `--max-block-size` is the maximum number of records any block may contain. If
  there are regions larger than this, the regions are broken down into chunks of
  this size; those chunks are _preassigned_ since they must be in a block all by
  themselves: we remove the chunks that are of this max size and remove a
  corresponding block from the ILP formulation. 
- `--num-blocks` is the number of blocks available to use (not all need be used
  by the ILP solution). If not set, it is chosen so that the average utlization
(number of points contained / max block size) is `--utilization`. 
- `--utilization` is the average block utilization of the solution (number of records
  / total block capacity). If not set, defaults to 0.5.


