// Package filter implements HDF5 filter decompression.
//
// Filters are applied to chunked data in reverse order during reading.
// Each filter transforms data from its encoded form to decoded form.
package filter

import (
	"fmt"

	"github.com/huangzhengshun/go-hdf5/internal/message"
)

// Filter is the interface implemented by all HDF5 filters.
type Filter interface {
	// ID returns the filter identifier.
	ID() uint16

	// Decode transforms encoded data to decoded form.
	Decode(input []byte) ([]byte, error)
}

// Registry maps filter IDs to filter constructors.
var Registry = map[uint16]func([]uint32) Filter{
	message.FilterDeflate:    func(cd []uint32) Filter { return NewDeflate(cd) },
	message.FilterShuffle:    func(cd []uint32) Filter { return NewShuffle(cd) },
	message.FilterFletcher32: func(cd []uint32) Filter { return NewFletcher32(cd) },
}

// filterNames maps known filter IDs to their names for better error messages.
var filterNames = map[uint16]string{
	message.FilterDeflate:     "deflate/gzip",
	message.FilterShuffle:     "shuffle",
	message.FilterFletcher32:  "Fletcher32",
	message.FilterSZIP:        "SZIP",
	message.FilterNBit:        "N-bit",
	message.FilterScaleOffset: "scale-offset",
}

// New creates a filter from a FilterInfo.
func New(info message.FilterInfo) (Filter, error) {
	constructor, ok := Registry[info.ID]
	if !ok {
		if info.IsOptional() {
			return nil, nil // Optional filter not available
		}
		// Provide helpful error message for known filters
		if name, known := filterNames[info.ID]; known {
			return nil, fmt.Errorf("%s filter (ID %d) is not supported; this dataset cannot be read", name, info.ID)
		}
		return nil, fmt.Errorf("unsupported filter ID: %d", info.ID)
	}
	return constructor(info.ClientData), nil
}
