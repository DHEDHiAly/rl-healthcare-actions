"""Streaming parquet loader for large tables."""
import pyarrow.parquet as pq
def stream_read(path, chunksize=100000):
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=chunksize):
        yield batch.to_pandas()
