package filter

import (
	"github.com/huangzhengshun/go-hdf5/internal/message"
)

// Shuffle implements the byte shuffle filter.
// This filter rearranges bytes to improve compression by grouping
// similar byte positions together (e.g., all MSBs, then all next bytes, etc.).
type Shuffle struct {
	elemSize int
}

// NewShuffle creates a new shuffle filter.
// Client data: [0] = element size in bytes
func NewShuffle(clientData []uint32) *Shuffle {
	elemSize := 1
	if len(clientData) > 0 && clientData[0] > 0 {
		elemSize = int(clientData[0])
	}
	return &Shuffle{elemSize: elemSize}
}

func (f *Shuffle) ID() uint16 {
	return message.FilterShuffle
}

// Decode reverses the shuffle transformation.
// Input is organized as: [all byte 0s][all byte 1s]...[all byte N-1s]
// Output is organized as: [elem0][elem1]...[elemM]
func (f *Shuffle) Decode(input []byte) ([]byte, error) {
	if f.elemSize <= 1 {
		// No shuffling for single-byte elements
		return input, nil
	}

	numBytes := len(input)
	numElems := numBytes / f.elemSize

	if numElems == 0 {
		return input, nil
	}

	output := make([]byte, numBytes)

	// Unshuffle: gather bytes from grouped positions into elements
	for i := 0; i < numElems; i++ {
		for j := 0; j < f.elemSize; j++ {
			// In shuffled format, byte j of all elements is at offset j*numElems
			output[i*f.elemSize+j] = input[j*numElems+i]
		}
	}

	return output, nil
}

// SetElementSize sets the element size for the shuffle filter.
// This is used when the element size is determined after filter creation.
func (f *Shuffle) SetElementSize(size int) {
	f.elemSize = size
}
