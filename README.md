### Data partitioning

## Requirements
You just need Python 3.x and the following modules:
```
pip3 install numpy progress
```

## Running

To run the ILP partitioning algorithm on a given query set:
```
python partition.py --query-dir --work-dir
    [ --timeout-sec ] [ --max-block-size ] [ --num-blocks]
```
Arguments:
- `--query-dir` is a directory containing only the query files: one file for each
query with the row IDs of the records returned by that query. Each file should
be in binary format, with each ID being an int32. It is unpacked with
`np.fromfile(.., dtype=np.int32)`
- `--work-dir` is a directory with intermediate results. All work dirs are
  subdirectories of the `scratch` directory since it is ignored by git. The work
dir is also used to cache some results so we don't have to invert the queries
every time.
- `--timeout-sec` is the timeout passed to the Gurobi optimizer. After this
  amount of time, the best solution found so far is returned. Note that this
timeout applies to _only_ the Gurobi ILP solver and not other components, like
query inversion.
- `--max-block-size` is the maximum number of records any block may contain.
  Currently, it must be larger than the size (number of points) of the largest region. If it is not
set, it is assumed to be exactly the number of points in the largest
intersection region.
- `--num-blocks` is the number of blocks available to use (not all need be used
  by the ILP solution). If not set, it is chosen so that the average utlization
(number of points contained / max block size) is about 0.5.

To generate sampled queries:
```
python dedup.py --source-dir=<original query dir> --output-dir=<output
query_dir> --sample=<sampling ratio>
```


