package layout

import (
	"fmt"

	"github.com/huangzhengshun/go-hdf5/internal/message"
)

// Compact represents compact storage layout.
// Data is stored directly in the object header.
type Compact struct {
	data      []byte
	dataspace *message.Dataspace
	datatype  *message.Datatype
}

// NewCompact creates a new compact layout handler.
func NewCompact(layout *message.DataLayout, dataspace *message.Dataspace, datatype *message.Datatype) *Compact {
	return &Compact{
		data:      layout.CompactData,
		dataspace: dataspace,
		datatype:  datatype,
	}
}

func (c *Compact) Class() message.LayoutClass {
	return message.LayoutCompact
}

// Read returns the compact data stored in the object header.
func (c *Compact) Read() ([]byte, error) {
	// Data is already available - just return a copy
	result := make([]byte, len(c.data))
	copy(result, c.data)
	return result, nil
}

// Size returns the size of the compact data.
func (c *Compact) Size() int {
	return len(c.data)
}

// ReadSlice reads a hyperslab from compact storage.
func (c *Compact) ReadSlice(start, count []uint64) ([]byte, error) {
	dims := c.dataspace.Dimensions
	if len(dims) == 0 {
		// Scalar dataset
		if len(start) == 0 && len(count) == 0 {
			result := make([]byte, len(c.data))
			copy(result, c.data)
			return result, nil
		}
		return nil, fmt.Errorf("cannot slice scalar dataset with non-empty start/count")
	}

	if len(start) != len(dims) || len(count) != len(dims) {
		return nil, fmt.Errorf("start and count must have %d dimensions, got %d and %d",
			len(dims), len(start), len(count))
	}

	// Validate bounds
	for d := 0; d < len(dims); d++ {
		if start[d]+count[d] > dims[d] {
			return nil, fmt.Errorf("slice out of bounds: dimension %d, start=%d, count=%d, size=%d",
				d, start[d], count[d], dims[d])
		}
	}

	elementSize := uint64(c.datatype.Size)
	return extractHyperslab(c.data, dims, start, count, elementSize)
}
