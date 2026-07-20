// Package object handles parsing of HDF5 object headers.
//
// Object headers contain metadata about HDF5 objects (groups, datasets, etc.)
// including dataspace, datatype, storage layout, and attributes.
package object

import (
	"errors"
	"fmt"

	"github.com/huangzhengshun/go-hdf5/internal/binary"
	"github.com/huangzhengshun/go-hdf5/internal/message"
)

// Object header signatures
var (
	SignatureV2 = []byte{'O', 'H', 'D', 'R'}
)

// Errors
var (
	ErrInvalidHeader        = errors.New("invalid object header")
	ErrUnsupportedVersion   = errors.New("unsupported object header version")
	ErrChecksumMismatch     = errors.New("object header checksum mismatch")
)

// Header represents a parsed HDF5 object header.
type Header struct {
	// Version is the object header version (1 or 2)
	Version uint8

	// Address is the file address where this header was found
	Address uint64

	// Flags contains header flags (v2 only)
	Flags uint8

	// RefCount is the reference count for this object
	RefCount uint32

	// Messages contains all parsed header messages
	Messages []message.Message

	// Timestamps (v2 only, if flag 0x04 is set)
	AccessTime uint32
	ModTime    uint32
	ChangeTime uint32
	BirthTime  uint32
}

// Read parses an object header at the given address.
func Read(r *binary.Reader, address uint64) (*Header, error) {
	hr := r.At(int64(address))

	// Peek first byte to determine version
	peek, err := hr.Peek(4)
	if err != nil {
		return nil, fmt.Errorf("reading object header: %w", err)
	}

	// Check for v2 signature "OHDR"
	if string(peek) == "OHDR" {
		return readV2(hr, address)
	}

	// Otherwise assume v1 (first byte is version number)
	if peek[0] == 1 {
		return readV1(hr, address)
	}

	return nil, fmt.Errorf("%w: unknown format at address %d", ErrInvalidHeader, address)
}

// GetMessage returns the first message of the given type, or nil if not found.
func (h *Header) GetMessage(typ message.Type) message.Message {
	for _, msg := range h.Messages {
		if msg.Type() == typ {
			return msg
		}
	}
	return nil
}

// GetMessages returns all messages of the given type.
func (h *Header) GetMessages(typ message.Type) []message.Message {
	var result []message.Message
	for _, msg := range h.Messages {
		if msg.Type() == typ {
			result = append(result, msg)
		}
	}
	return result
}

// Dataspace returns the dataspace message if present.
func (h *Header) Dataspace() *message.Dataspace {
	msg := h.GetMessage(message.TypeDataspace)
	if msg == nil {
		return nil
	}
	return msg.(*message.Dataspace)
}

// Datatype returns the datatype message if present.
func (h *Header) Datatype() *message.Datatype {
	msg := h.GetMessage(message.TypeDatatype)
	if msg == nil {
		return nil
	}
	return msg.(*message.Datatype)
}

// DataLayout returns the data layout message if present.
func (h *Header) DataLayout() *message.DataLayout {
	msg := h.GetMessage(message.TypeDataLayout)
	if msg == nil {
		return nil
	}
	return msg.(*message.DataLayout)
}

// FilterPipeline returns the filter pipeline message if present.
func (h *Header) FilterPipeline() *message.FilterPipeline {
	msg := h.GetMessage(message.TypeFilterPipeline)
	if msg == nil {
		return nil
	}
	return msg.(*message.FilterPipeline)
}
