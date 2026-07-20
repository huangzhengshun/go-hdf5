package message

import (
	"encoding/binary"
	"fmt"

	binpkg "github.com/huangzhengshun/go-hdf5/internal/binary"
)

// DataspaceType represents the type of dataspace.
type DataspaceType uint8

const (
	DataspaceScalar DataspaceType = 0 // Single element
	DataspaceSimple DataspaceType = 1 // Regular N-dimensional array
	DataspaceNull   DataspaceType = 2 // No data
)

// Dataspace represents a dataspace message (type 0x0001).
type Dataspace struct {
	Version    uint8
	Rank       int
	SpaceType  DataspaceType
	Dimensions []uint64
	MaxDims    []uint64 // nil if not present (means same as Dimensions)
}

func (m *Dataspace) Type() Type { return TypeDataspace }

// NumElements returns the total number of elements in the dataspace.
func (m *Dataspace) NumElements() uint64 {
	switch m.SpaceType {
	case DataspaceNull:
		return 0
	case DataspaceScalar:
		return 1
	case DataspaceSimple:
		if len(m.Dimensions) == 0 {
			return 0
		}
		n := uint64(1)
		for _, d := range m.Dimensions {
			n *= d
		}
		return n
	default:
		return 0
	}
}

// IsScalar returns true if this is a scalar dataspace.
func (m *Dataspace) IsScalar() bool {
	return m.SpaceType == DataspaceScalar
}

// IsNull returns true if this is a null dataspace.
func (m *Dataspace) IsNull() bool {
	return m.SpaceType == DataspaceNull
}

func parseDataspace(data []byte, r *binpkg.Reader) (*Dataspace, error) {
	if len(data) < 4 {
		return nil, fmt.Errorf("dataspace message too short")
	}

	ds := &Dataspace{
		Version: data[0],
		Rank:    int(data[1]),
	}

	flags := data[2]
	hasMaxDims := flags&0x01 != 0

	// Version 2 has explicit type field
	if ds.Version >= 2 {
		ds.SpaceType = DataspaceType(data[3])
	} else {
		// Version 1: infer type from rank
		if ds.Rank == 0 {
			ds.SpaceType = DataspaceScalar
		} else {
			ds.SpaceType = DataspaceSimple
		}
	}

	// No dimensions for scalar or null
	if ds.SpaceType != DataspaceSimple || ds.Rank == 0 {
		return ds, nil
	}

	// Calculate offset to dimensions
	offset := 4
	if ds.Version == 1 {
		offset = 8 // Version 1 has 4 reserved bytes
	}

	// Use the reader's length size for dimension values
	lengthSize := r.LengthSize()
	if lengthSize == 0 {
		lengthSize = 8 // Default to 8 bytes
	}

	// Parse dimensions
	ds.Dimensions = make([]uint64, ds.Rank)
	for i := 0; i < ds.Rank; i++ {
		if offset+lengthSize > len(data) {
			return nil, fmt.Errorf("dataspace message truncated reading dimensions")
		}
		ds.Dimensions[i] = decodeUint(data[offset:], lengthSize, r.ByteOrder())
		offset += lengthSize
	}

	// Parse max dimensions if present
	if hasMaxDims {
		ds.MaxDims = make([]uint64, ds.Rank)
		for i := 0; i < ds.Rank; i++ {
			if offset+lengthSize > len(data) {
				return nil, fmt.Errorf("dataspace message truncated reading max dimensions")
			}
			ds.MaxDims[i] = decodeUint(data[offset:], lengthSize, r.ByteOrder())
			offset += lengthSize
		}
	}

	return ds, nil
}

// decodeUint decodes a variable-width unsigned integer.
func decodeUint(buf []byte, size int, order binary.ByteOrder) uint64 {
	switch size {
	case 1:
		return uint64(buf[0])
	case 2:
		return uint64(order.Uint16(buf))
	case 4:
		return uint64(order.Uint32(buf))
	case 8:
		return order.Uint64(buf)
	default:
		var val uint64
		for i := size - 1; i >= 0; i-- {
			val = (val << 8) | uint64(buf[i])
		}
		return val
	}
}
