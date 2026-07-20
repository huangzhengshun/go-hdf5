package message

import (
	"encoding/binary"
	"fmt"

	binpkg "github.com/huangzhengshun/go-hdf5/internal/binary"
)

// LayoutClass represents the storage layout class.
type LayoutClass uint8

const (
	LayoutCompact    LayoutClass = 0 // Data stored in object header
	LayoutContiguous LayoutClass = 1 // Data in single contiguous block
	LayoutChunked    LayoutClass = 2 // Data in indexed chunks
	LayoutVirtual    LayoutClass = 3 // Virtual dataset (v4+)
)

// ChunkIndexType represents the type of chunk index used in v3/v4 layouts.
// Values match HDF5 spec: stored in bits 0-4 of the flags byte.
type ChunkIndexType uint8

const (
	ChunkIndexSingleChunk     ChunkIndexType = 1 // Single chunk (no index needed)
	ChunkIndexImplicit        ChunkIndexType = 2 // Implicit (contiguous chunks)
	ChunkIndexFixedArray      ChunkIndexType = 3 // Fixed array
	ChunkIndexExtensibleArray ChunkIndexType = 4 // Extensible array
	ChunkIndexBTreeV2         ChunkIndexType = 5 // B-tree v2
)

// DataLayout represents a data layout message (type 0x0008).
type DataLayout struct {
	Version uint8
	Class   LayoutClass

	// Compact layout: data is stored directly
	CompactData []byte

	// Contiguous layout
	Address uint64 // Address of data
	Size    uint64 // Size of data in bytes

	// Chunked layout
	ChunkDims      []uint32       // Size of each chunk dimension
	ChunkIndexAddr uint64         // Address of B-tree (v1/v2) or chunk index
	ChunkIndexType ChunkIndexType // Type of chunk index (v3/v4 only)

	// Chunked layout v3+ additional fields
	ChunkFlags         uint8
	DimensionSizeBytes uint8 // Size of each dimension entry

	// Filtered chunk info (v4)
	FilteredChunkSize uint32
}

func (m *DataLayout) Type() Type { return TypeDataLayout }

// IsCompact returns true if data is stored in the object header.
func (m *DataLayout) IsCompact() bool {
	return m.Class == LayoutCompact
}

// IsContiguous returns true if data is stored contiguously.
func (m *DataLayout) IsContiguous() bool {
	return m.Class == LayoutContiguous
}

// IsChunked returns true if data is stored in chunks.
func (m *DataLayout) IsChunked() bool {
	return m.Class == LayoutChunked
}

func parseDataLayout(data []byte, r *binpkg.Reader) (*DataLayout, error) {
	if len(data) < 2 {
		return nil, fmt.Errorf("data layout message too short")
	}

	layout := &DataLayout{
		Version: data[0],
	}

	switch layout.Version {
	case 1, 2:
		return parseDataLayoutV1V2(data, r, layout)
	case 3:
		return parseDataLayoutV3(data, r, layout)
	case 4:
		return parseDataLayoutV4(data, r, layout)
	default:
		return nil, fmt.Errorf("unsupported data layout version: %d", layout.Version)
	}
}

func parseDataLayoutV1V2(data []byte, r *binpkg.Reader, layout *DataLayout) (*DataLayout, error) {
	if len(data) < 4 {
		return nil, fmt.Errorf("data layout v1/v2 message too short")
	}

	ndims := int(data[1])
	layout.Class = LayoutClass(data[2])
	// data[3] is reserved

	offset := 4

	switch layout.Class {
	case LayoutCompact:
		if offset+4 > len(data) {
			return nil, fmt.Errorf("compact layout truncated")
		}
		size := binary.LittleEndian.Uint32(data[offset:])
		offset += 4
		if offset+int(size) > len(data) {
			return nil, fmt.Errorf("compact data truncated")
		}
		layout.CompactData = make([]byte, size)
		copy(layout.CompactData, data[offset:offset+int(size)])

	case LayoutContiguous:
		offsetSize := r.OffsetSize()
		lengthSize := r.LengthSize()
		if offset+offsetSize+lengthSize > len(data) {
			return nil, fmt.Errorf("contiguous layout truncated")
		}
		layout.Address = decodeUint(data[offset:], offsetSize, r.ByteOrder())
		offset += offsetSize
		layout.Size = decodeUint(data[offset:], lengthSize, r.ByteOrder())

	case LayoutChunked:
		offsetSize := r.OffsetSize()
		if offset+offsetSize > len(data) {
			return nil, fmt.Errorf("chunked layout truncated")
		}
		layout.ChunkIndexAddr = decodeUint(data[offset:], offsetSize, r.ByteOrder())
		offset += offsetSize

		// Parse chunk dimensions (ndims * 4 bytes each)
		layout.ChunkDims = make([]uint32, ndims)
		for i := 0; i < ndims && offset+4 <= len(data); i++ {
			layout.ChunkDims[i] = binary.LittleEndian.Uint32(data[offset:])
			offset += 4
		}
	}

	return layout, nil
}

func parseDataLayoutV3(data []byte, r *binpkg.Reader, layout *DataLayout) (*DataLayout, error) {
	if len(data) < 2 {
		return nil, fmt.Errorf("data layout v3 message too short")
	}

	layout.Class = LayoutClass(data[1])
	offset := 2

	switch layout.Class {
	case LayoutCompact:
		if offset+2 > len(data) {
			return nil, fmt.Errorf("compact layout v3 truncated")
		}
		size := binary.LittleEndian.Uint16(data[offset:])
		offset += 2
		if offset+int(size) > len(data) {
			return nil, fmt.Errorf("compact data v3 truncated")
		}
		layout.CompactData = make([]byte, size)
		copy(layout.CompactData, data[offset:offset+int(size)])

	case LayoutContiguous:
		offsetSize := r.OffsetSize()
		lengthSize := r.LengthSize()
		if offset+offsetSize+lengthSize > len(data) {
			return nil, fmt.Errorf("contiguous layout v3 truncated")
		}
		layout.Address = decodeUint(data[offset:], offsetSize, r.ByteOrder())
		offset += offsetSize
		layout.Size = decodeUint(data[offset:], lengthSize, r.ByteOrder())

	case LayoutChunked:
		if offset+2 > len(data) {
			return nil, fmt.Errorf("chunked layout v3 truncated")
		}
		layout.ChunkFlags = data[offset]
		layout.ChunkIndexType = ChunkIndexType(layout.ChunkFlags & 0x0F) // Lower 4 bits
		offset++
		ndims := int(data[offset])
		offset++
		layout.DimensionSizeBytes = data[offset]
		offset++

		// Parse chunk dimensions
		dimSize := int(layout.DimensionSizeBytes)
		layout.ChunkDims = make([]uint32, ndims)
		for i := 0; i < ndims && offset+dimSize <= len(data); i++ {
			layout.ChunkDims[i] = uint32(decodeUint(data[offset:], dimSize, r.ByteOrder()))
			offset += dimSize
		}

		// For v4, there may be additional bytes before the address
		// (element size encoding or chunk indexing type parameters)
		// The address is typically at the end of the message, so we calculate its position
		offsetSize := r.OffsetSize()
		addrOffset := len(data) - offsetSize
		if addrOffset >= offset {
			layout.ChunkIndexAddr = decodeUint(data[addrOffset:], offsetSize, r.ByteOrder())
		} else if offset+offsetSize <= len(data) {
			layout.ChunkIndexAddr = decodeUint(data[offset:], offsetSize, r.ByteOrder())
		}
	}

	return layout, nil
}

func parseDataLayoutV4(data []byte, r *binpkg.Reader, layout *DataLayout) (*DataLayout, error) {
	// V4 is similar to V3 but supports virtual datasets
	// For now, parse like V3
	return parseDataLayoutV3(data, r, layout)
}
