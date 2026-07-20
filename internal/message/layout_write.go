package message

import (
	"github.com/huangzhengshun/go-hdf5/internal/binary"
)

// Serialize writes the DataLayout to the writer.
// Uses version 3/4 format for modern compatibility.
func (m *DataLayout) Serialize(w *binary.Writer) error {
	// We use version 4 for chunked with index type, version 3 otherwise
	version := m.Version
	if version == 0 {
		if m.Class == LayoutChunked {
			version = 4
		} else {
			version = 3
		}
	}

	if err := w.WriteUint8(version); err != nil {
		return err
	}

	if err := w.WriteUint8(uint8(m.Class)); err != nil {
		return err
	}

	switch m.Class {
	case LayoutCompact:
		// Version 3: 2-byte size
		if err := w.WriteUint16(uint16(len(m.CompactData))); err != nil {
			return err
		}
		if err := w.WriteBytes(m.CompactData); err != nil {
			return err
		}

	case LayoutContiguous:
		// Address + size
		if err := w.WriteOffset(m.Address); err != nil {
			return err
		}
		if err := w.WriteLength(m.Size); err != nil {
			return err
		}

	case LayoutChunked:
		// Flags byte (bit 4 = DONT_FILTER_PARTIAL_BOUND_CHUNKS)
		// Note: chunk index type is written as a separate byte after dimensions
		flags := uint8(0)
		if err := w.WriteUint8(flags); err != nil {
			return err
		}

		// Number of dimensions (including element size as extra dimension)
		ndims := uint8(len(m.ChunkDims))
		if err := w.WriteUint8(ndims); err != nil {
			return err
		}

		// Dimension size bytes (calculate from chunk dims)
		dimSizeBytes := m.DimensionSizeBytes
		if dimSizeBytes == 0 {
			dimSizeBytes = 4 // Default to 4 bytes
		}
		if err := w.WriteUint8(dimSizeBytes); err != nil {
			return err
		}

		// Chunk dimensions (includes element size as last dimension)
		for _, dim := range m.ChunkDims {
			if err := w.WriteUintN(uint64(dim), int(dimSizeBytes)); err != nil {
				return err
			}
		}

		// Chunk Index Type (separate byte for v4 layout)
		if err := w.WriteUint8(uint8(m.ChunkIndexType)); err != nil {
			return err
		}

		// Indexing-type-specific info (required for Fixed Array and Extensible Array)
		switch m.ChunkIndexType {
		case ChunkIndexFixedArray:
			// Page Bits: log2 of entries per data block page
			// Must match the value used in WriteFixedArrayIndex
			pageBits := uint8(10) // Match h5py and our FAHD
			if err := w.WriteUint8(pageBits); err != nil {
				return err
			}
		case ChunkIndexExtensibleArray:
			// Max Bits: max number of bits for storing # of elements in data block page
			// h5py uses 10 for small datasets
			maxBits := uint8(10)
			if err := w.WriteUint8(maxBits); err != nil {
				return err
			}
		}

		// Write index address
		if err := w.WriteOffset(m.ChunkIndexAddr); err != nil {
			return err
		}
	}

	return nil
}

// SerializedSize returns the size in bytes when serialized.
func (m *DataLayout) SerializedSize(w *binary.Writer) int {
	// Version + class
	size := 2

	switch m.Class {
	case LayoutCompact:
		size += 2 + len(m.CompactData)

	case LayoutContiguous:
		size += w.OffsetSize() + w.LengthSize()

	case LayoutChunked:
		dimSizeBytes := int(m.DimensionSizeBytes)
		if dimSizeBytes == 0 {
			dimSizeBytes = 4
		}
		// flags(1) + ndims(1) + dimSizeBytes(1) + dims(ndims*dimSizeBytes) + indexType(1) + indexInfo(1 if needed) + indexAddr(offsetSize)
		size += 3
		size += len(m.ChunkDims) * dimSizeBytes
		size += 1 // chunk index type (separate byte)
		// Indexing-type-specific info byte (Fixed Array and Extensible Array need this)
		if m.ChunkIndexType == ChunkIndexFixedArray || m.ChunkIndexType == ChunkIndexExtensibleArray {
			size += 1
		}
		size += w.OffsetSize() // chunk index address
	}

	return size
}

// NewCompactLayout creates a new compact layout message.
func NewCompactLayout(data []byte) *DataLayout {
	return &DataLayout{
		Version:     3,
		Class:       LayoutCompact,
		CompactData: data,
	}
}

// NewContiguousLayout creates a new contiguous layout message.
// Address and Size will be set later when data is written.
func NewContiguousLayout(address, size uint64) *DataLayout {
	return &DataLayout{
		Version: 3,
		Class:   LayoutContiguous,
		Address: address,
		Size:    size,
	}
}

// NewChunkedLayout creates a new chunked layout message.
// ChunkIndexAddr will be set later when the index is written.
// Note: In HDF5 v4 layout, chunk dimensions include an extra dimension for element size.
// The chunkDims passed should be the user-facing chunk dimensions (without element size).
// elementSize is the size of each element in bytes (e.g., 4 for int32).
func NewChunkedLayout(chunkDims []uint32, elementSize uint32, indexType ChunkIndexType) *DataLayout {
	// HDF5 spec says: "The number of elements in the Chunk Dimension Sizes field is one
	// greater than the number of dimensions in the dataset's dataspace"
	// The extra dimension is the element size in bytes.
	allDims := make([]uint32, len(chunkDims)+1)
	copy(allDims, chunkDims)
	allDims[len(chunkDims)] = elementSize

	// Choose optimal dimension size encoding
	var dimSizeBytes uint8 = 1
	for _, d := range allDims {
		if d > 0xFF && dimSizeBytes < 2 {
			dimSizeBytes = 2
		}
		if d > 0xFFFF && dimSizeBytes < 4 {
			dimSizeBytes = 4
		}
	}

	return &DataLayout{
		Version:            4,
		Class:              LayoutChunked,
		ChunkDims:          allDims,
		ChunkIndexType:     indexType,
		DimensionSizeBytes: dimSizeBytes,
	}
}
