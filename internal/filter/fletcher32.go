package filter

import (
	"encoding/binary"
	"fmt"

	binpkg "github.com/huangzhengshun/go-hdf5/internal/binary"
	"github.com/huangzhengshun/go-hdf5/internal/message"
)

// Fletcher32Filter implements the Fletcher-32 checksum filter.
// This filter validates data integrity by checking a checksum
// appended to the data.
type Fletcher32Filter struct{}

// NewFletcher32 creates a new Fletcher-32 filter.
func NewFletcher32(clientData []uint32) *Fletcher32Filter {
	return &Fletcher32Filter{}
}

func (f *Fletcher32Filter) ID() uint16 {
	return message.FilterFletcher32
}

// Decode verifies the Fletcher-32 checksum and returns the data without it.
// The checksum is stored as the last 4 bytes of the input.
func (f *Fletcher32Filter) Decode(input []byte) ([]byte, error) {
	if len(input) < 4 {
		return nil, fmt.Errorf("fletcher32: input too short for checksum")
	}

	// Split data and checksum
	data := input[:len(input)-4]
	checksumBytes := input[len(input)-4:]

	// Stored checksum (little-endian)
	storedChecksum := binary.LittleEndian.Uint32(checksumBytes)

	// Compute checksum of data
	computedChecksum := binpkg.Fletcher32(data)

	if storedChecksum != computedChecksum {
		return nil, fmt.Errorf("fletcher32: checksum mismatch (stored=0x%08x, computed=0x%08x)",
			storedChecksum, computedChecksum)
	}

	return data, nil
}
