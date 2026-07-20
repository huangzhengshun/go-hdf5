package layout

import (
	"fmt"

	"github.com/huangzhengshun/go-hdf5/internal/binary"
	"github.com/huangzhengshun/go-hdf5/internal/message"
)

// Contiguous represents contiguous storage layout.
// Data is stored in a single contiguous block in the file.
type Contiguous struct {
	address   uint64
	size      uint64
	dataspace *message.Dataspace
	datatype  *message.Datatype
	reader    *binary.Reader
}

// NewContiguous creates a new contiguous layout handler.
func NewContiguous(
	layout *message.DataLayout,
	dataspace *message.Dataspace,
	datatype *message.Datatype,
	reader *binary.Reader,
) *Contiguous {
	size := layout.Size
	if size == 0 {
		// Calculate size from dataspace and datatype
		size = calculateDataSize(dataspace, datatype)
	}

	return &Contiguous{
		address:   layout.Address,
		size:      size,
		dataspace: dataspace,
		datatype:  datatype,
		reader:    reader,
	}
}

func (c *Contiguous) Class() message.LayoutClass {
	return message.LayoutContiguous
}

// Read reads all data from contiguous storage.
func (c *Contiguous) Read() ([]byte, error) {
	// Check for undefined address (no data allocated)
	if c.reader.IsUndefinedOffset(c.address) {
		return nil, fmt.Errorf("contiguous data not allocated")
	}

	if c.size == 0 {
		return []byte{}, nil
	}

	// Read data directly from the file
	r := c.reader.At(int64(c.address))
	data, err := r.ReadBytes(int(c.size))
	if err != nil {
		return nil, fmt.Errorf("reading contiguous data: %w", err)
	}

	return data, nil
}

// Address returns the data address.
func (c *Contiguous) Address() uint64 {
	return c.address
}

// Size returns the data size in bytes.
func (c *Contiguous) Size() uint64 {
	return c.size
}

// ReadSlice reads a hyperslab from contiguous storage.
func (c *Contiguous) ReadSlice(start, count []uint64) ([]byte, error) {
	dims := c.dataspace.Dimensions
	if len(dims) == 0 {
		// Scalar dataset
		if len(start) == 0 && len(count) == 0 {
			return c.Read()
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
	ndims := len(dims)

	// For 1D arrays or when selecting a contiguous region, we can optimize
	if ndims == 1 {
		// Simple case: read only the needed portion
		startByte := start[0] * elementSize
		numBytes := count[0] * elementSize
		r := c.reader.At(int64(c.address + startByte))
		return r.ReadBytes(int(numBytes))
	}

	// For multi-dimensional arrays, check if we can read a contiguous block
	// This is possible when the selection spans complete rows in all but the first dimension
	canOptimize := true
	for d := 1; d < ndims; d++ {
		if start[d] != 0 || count[d] != dims[d] {
			canOptimize = false
			break
		}
	}

	if canOptimize {
		// Calculate row size (all elements in dims[1:])
		rowSize := elementSize
		for d := 1; d < ndims; d++ {
			rowSize *= dims[d]
		}
		startByte := start[0] * rowSize
		numBytes := count[0] * rowSize
		r := c.reader.At(int64(c.address + startByte))
		return r.ReadBytes(int(numBytes))
	}

	// General case: read all data and extract the hyperslab
	// This is less efficient but handles all cases
	data, err := c.Read()
	if err != nil {
		return nil, err
	}

	return extractHyperslab(data, dims, start, count, elementSize)
}
