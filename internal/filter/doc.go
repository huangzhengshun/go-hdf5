// Package filter implements the HDF5 filter pipeline for data decompression.
//
// HDF5 uses filters to compress and transform chunked data. When reading,
// filters are applied in reverse order to decode the data. Each chunk can
// have a filter mask that allows skipping specific filters.
//
// # Supported Filters
//
// This package implements the following standard HDF5 filters:
//
//   - DEFLATE (ID 1): Zlib/gzip compression via [Deflate]. This is the most
//     common compression filter, using Go's standard compress/zlib package.
//
//   - Shuffle (ID 2): Byte shuffling via [Shuffle]. Rearranges bytes to
//     improve compression by grouping similar byte positions together
//     (e.g., all MSBs, then all second bytes, etc.).
//
//   - Fletcher32 (ID 3): Checksum validation via [Fletcher32Filter]. Verifies
//     data integrity by checking a 32-bit Fletcher checksum appended to
//     the data.
//
// # Unsupported Filters
//
// The following filters are recognized but not implemented:
//
//   - SZIP (ID 4): Proprietary compression algorithm
//   - N-bit (ID 5): Bit-level packing
//   - Scale-offset (ID 6): Integer scaling and offset
//
// Datasets using unsupported filters cannot be read. However, optional filters
// (marked in the filter pipeline) can be skipped if not available.
//
// # Filter Pipeline
//
// The [Pipeline] type manages a sequence of filters for a dataset:
//
//	pipeline, err := filter.NewPipeline(filterPipelineMsg)
//	decodedData, err := pipeline.Decode(compressedData, filterMask)
//
// Filters are applied in reverse order during decoding. For example, if a
// dataset was written with filters [Shuffle, DEFLATE], decoding applies
// DEFLATE first (to decompress), then Shuffle (to unshuffle bytes).
//
// # Filter Mask
//
// Each chunk can have a filter mask that indicates which filters to skip.
// If bit i is set in the mask, filter i is skipped during decoding. This
// allows individual chunks to use different filter combinations.
//
// # Key Types
//
//   - [Filter]: Interface implemented by all filters (ID and Decode methods)
//   - [Pipeline]: Manages a sequence of filters for decoding
//   - [Deflate]: DEFLATE/zlib decompression filter
//   - [Shuffle]: Byte shuffle/unshuffle filter
//   - [Fletcher32Filter]: Fletcher-32 checksum verification filter
package filter
