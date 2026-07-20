// Package layout provides storage layout handlers for reading HDF5 dataset data.
//
// HDF5 datasets can store their raw data using three different storage layouts,
// each optimized for different use cases. This package provides a unified
// [Layout] interface for reading data regardless of the underlying storage
// mechanism.
//
// # Storage Layouts
//
// HDF5 defines three storage layout classes:
//
//   - Compact (class 0): Data is stored directly within the object header.
//     Best for small datasets where the overhead of external storage would
//     exceed the data size. Implemented by [Compact].
//
//   - Contiguous (class 1): Data is stored in a single contiguous block
//     in the file. Best for datasets that are written once and read
//     sequentially. Implemented by [Contiguous].
//
//   - Chunked (class 2): Data is divided into fixed-size chunks that are
//     stored separately and indexed by a B-tree or array structure.
//     Required for datasets with compression or extensible dimensions.
//     Implemented by [Chunked].
//
// # Reading Data
//
// Use [New] to create the appropriate layout handler:
//
//	layout, err := layout.New(layoutMsg, dataspaceMsg, datatypeMsg, filterPipelineMsg, reader)
//	data, err := layout.Read()
//
// # Chunked Storage Details
//
// Chunked storage supports multiple index formats, automatically detected:
//
//   - Single chunk: Small datasets with one chunk, no index structure
//   - B-tree v1 ("TREE"): Traditional B-tree index for chunks
//   - B-tree v2 ("BTHD"): Modern B-tree with better performance
//   - Fixed array ("FAHD"): Fixed-size array for known chunk counts
//   - Extensible array ("EAHD"): Growable array for extensible datasets
//
// The [Chunked] type handles decompression through the filter pipeline and
// correctly assembles chunks into the final dataset array, handling edge
// chunks that may be smaller than the chunk dimensions.
//
// # Multi-dimensional Chunk Copying
//
// For multi-dimensional datasets, the copyChunkRecursive algorithm handles
// copying chunk data to the correct positions in the output buffer. It works
// by recursively iterating through dimensions:
//
//  1. For each position in the current dimension, calculate the output and
//     chunk buffer offsets
//  2. Recurse to the next dimension until reaching the innermost dimension
//  3. At the innermost dimension, perform a contiguous memory copy
//
// This approach correctly handles partial edge chunks where the chunk
// extends beyond the dataset boundaries.
//
// # Key Types
//
//   - [Layout]: Interface for reading dataset data (Read and Class methods)
//   - [Compact]: Handler for compact storage (data in header)
//   - [Contiguous]: Handler for contiguous storage (single block)
//   - [Chunked]: Handler for chunked storage (indexed chunks with filters)
package layout
