package message

import (
	"github.com/huangzhengshun/go-hdf5/internal/binary"
)

// Serialize writes the Dataspace to the writer.
// Uses version 2 format for maximum compatibility.
func (m *Dataspace) Serialize(w *binary.Writer) error {
	// Version 2 format:
	// Byte 0: Version (2)
	// Byte 1: Dimensionality (rank)
	// Byte 2: Flags (bit 0 = max dims present)
	// Byte 3: Type (0=scalar, 1=simple, 2=null)
	// Followed by: dimensions (rank * lengthSize bytes each)
	// Followed by: max dimensions if present (rank * lengthSize bytes each)

	if err := w.WriteUint8(2); err != nil { // Version 2
		return err
	}

	if err := w.WriteUint8(uint8(m.Rank)); err != nil {
		return err
	}

	flags := uint8(0)
	if m.MaxDims != nil && len(m.MaxDims) > 0 {
		flags |= 0x01
	}
	if err := w.WriteUint8(flags); err != nil {
		return err
	}

	if err := w.WriteUint8(uint8(m.SpaceType)); err != nil {
		return err
	}

	// Write dimensions
	for _, dim := range m.Dimensions {
		if err := w.WriteLength(dim); err != nil {
			return err
		}
	}

	// Write max dimensions if present
	if m.MaxDims != nil {
		for _, maxDim := range m.MaxDims {
			if err := w.WriteLength(maxDim); err != nil {
				return err
			}
		}
	}

	return nil
}

// SerializedSize returns the size in bytes when serialized.
func (m *Dataspace) SerializedSize(w *binary.Writer) int {
	// Header: 4 bytes (version, rank, flags, type)
	size := 4

	// Dimensions: rank * lengthSize
	size += m.Rank * w.LengthSize()

	// Max dimensions: rank * lengthSize (if present)
	if m.MaxDims != nil && len(m.MaxDims) > 0 {
		size += m.Rank * w.LengthSize()
	}

	return size
}

// NewDataspace creates a new Dataspace message for simple datasets.
func NewDataspace(dims []uint64, maxDims []uint64) *Dataspace {
	ds := &Dataspace{
		Version:    2,
		Rank:       len(dims),
		SpaceType:  DataspaceSimple,
		Dimensions: dims,
	}

	if maxDims != nil {
		ds.MaxDims = maxDims
	}

	return ds
}

// NewScalarDataspace creates a new scalar Dataspace message.
func NewScalarDataspace() *Dataspace {
	return &Dataspace{
		Version:   2,
		Rank:      0,
		SpaceType: DataspaceScalar,
	}
}

// NewNullDataspace creates a new null Dataspace message.
func NewNullDataspace() *Dataspace {
	return &Dataspace{
		Version:   2,
		Rank:      0,
		SpaceType: DataspaceNull,
	}
}
