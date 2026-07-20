package message

import (
	"github.com/huangzhengshun/go-hdf5/internal/binary"
)

// LinkInfo represents a link info message (type 0x0002).
// This message provides metadata about links in a group.
type LinkInfo struct {
	Version               uint8
	Flags                 uint8
	MaxCreationIndex      uint64 // Present if flag bit 0 set
	FractalHeapAddr       uint64 // Present if flag bit 1 set
	NameIndexBTreeAddr    uint64 // Present if flag bit 1 set
	CreationOrderBTreeAddr uint64 // Present if both bits 0 and 1 set
}

func (m *LinkInfo) Type() Type { return TypeLinkInfo }

// Serialize writes the LinkInfo to the writer.
// Note: The HDF5 library expects fractal heap and B-tree addresses to always
// be present for groups using Link messages, even when undefined.
func (m *LinkInfo) Serialize(w *binary.Writer) error {
	if err := w.WriteUint8(m.Version); err != nil {
		return err
	}

	if err := w.WriteUint8(m.Flags); err != nil {
		return err
	}

	// Maximum creation index (if flag bit 0 set)
	if m.Flags&0x01 != 0 {
		if err := w.WriteUint64(m.MaxCreationIndex); err != nil {
			return err
		}
	}

	// Fractal heap and name index B-tree addresses
	// The HDF5 library seems to always write these, using 0xFFFFFFFFFFFFFFFF for undefined
	if err := w.WriteOffset(m.FractalHeapAddr); err != nil {
		return err
	}
	if err := w.WriteOffset(m.NameIndexBTreeAddr); err != nil {
		return err
	}

	// Creation order B-tree address (if both bits 0 and 1 set)
	if m.Flags&0x03 == 0x03 {
		if err := w.WriteOffset(m.CreationOrderBTreeAddr); err != nil {
			return err
		}
	}

	return nil
}

// SerializedSize returns the size in bytes when serialized.
func (m *LinkInfo) SerializedSize(w *binary.Writer) int {
	// Version + flags
	size := 2

	// Maximum creation index
	if m.Flags&0x01 != 0 {
		size += 8
	}

	// Fractal heap + name index B-tree addresses (always present)
	size += 2 * w.OffsetSize()

	// Creation order B-tree address
	if m.Flags&0x03 == 0x03 {
		size += w.OffsetSize()
	}

	return size
}

// UndefinedAddress is the HDF5 undefined address value.
const UndefinedAddress = ^uint64(0)

// NewLinkInfo creates a new minimal LinkInfo message.
// This creates an empty link info with undefined heap/B-tree addresses.
func NewLinkInfo() *LinkInfo {
	return &LinkInfo{
		Version:            0,
		Flags:              0,
		FractalHeapAddr:    UndefinedAddress,
		NameIndexBTreeAddr: UndefinedAddress,
	}
}

// NewLinkInfoWithHeap creates a LinkInfo message that uses a fractal heap.
// Use undefined addresses (0xFFFFFFFFFFFFFFFF) when no heap exists yet.
func NewLinkInfoWithHeap(heapAddr, nameIndexAddr uint64) *LinkInfo {
	return &LinkInfo{
		Version:            0,
		Flags:              0x02, // Use fractal heap
		FractalHeapAddr:    heapAddr,
		NameIndexBTreeAddr: nameIndexAddr,
	}
}
