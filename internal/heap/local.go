// Package heap implements HDF5 heap structures.
package heap

import (
	"fmt"

	"github.com/huangzhengshun/go-hdf5/internal/binary"
)

// LocalHeap represents an HDF5 local heap for storing variable-length data
// (typically names in v1 groups).
type LocalHeap struct {
	DataSize    uint64
	FreeOffset  uint64
	DataAddress uint64
	data        []byte
}

// Signature for local heap: "HEAP"
var localHeapSignature = []byte{'H', 'E', 'A', 'P'}

// ReadLocalHeap reads a local heap at the given address.
func ReadLocalHeap(r *binary.Reader, address uint64) (*LocalHeap, error) {
	hr := r.At(int64(address))

	// Check signature
	sig, err := hr.ReadBytes(4)
	if err != nil {
		return nil, fmt.Errorf("reading local heap signature: %w", err)
	}
	if string(sig) != "HEAP" {
		return nil, fmt.Errorf("invalid local heap signature: got %q, expected \"HEAP\"", string(sig))
	}

	// Version (1 byte)
	version, err := hr.ReadUint8()
	if err != nil {
		return nil, err
	}
	if version != 0 {
		return nil, fmt.Errorf("unsupported local heap version: %d", version)
	}

	// Reserved (3 bytes)
	hr.Skip(3)

	// Data segment size (length-sized)
	dataSize, err := hr.ReadLength()
	if err != nil {
		return nil, err
	}

	// Offset to head of free list (length-sized)
	freeOffset, err := hr.ReadLength()
	if err != nil {
		return nil, err
	}

	// Data segment address (offset-sized)
	dataAddr, err := hr.ReadOffset()
	if err != nil {
		return nil, err
	}

	heap := &LocalHeap{
		DataSize:    dataSize,
		FreeOffset:  freeOffset,
		DataAddress: dataAddr,
	}

	// Read the actual data segment
	dr := r.At(int64(dataAddr))
	heap.data, err = dr.ReadBytes(int(dataSize))
	if err != nil {
		return nil, fmt.Errorf("reading local heap data: %w", err)
	}

	return heap, nil
}

// GetString reads a null-terminated string at the given offset in the heap.
func (h *LocalHeap) GetString(offset uint64) string {
	if offset >= uint64(len(h.data)) {
		return ""
	}

	// Find null terminator
	end := offset
	for end < uint64(len(h.data)) && h.data[end] != 0 {
		end++
	}

	return string(h.data[offset:end])
}
