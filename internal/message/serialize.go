package message

import (
	"github.com/huangzhengshun/go-hdf5/internal/binary"
)

// Serializable is the interface for messages that can be serialized to bytes.
type Serializable interface {
	Message
	// Serialize writes the message to the writer.
	Serialize(w *binary.Writer) error
	// SerializedSize returns the size in bytes when serialized.
	SerializedSize(w *binary.Writer) int
}

// Serialize serializes a message if it implements Serializable.
func Serialize(msg Message, w *binary.Writer) error {
	if s, ok := msg.(Serializable); ok {
		return s.Serialize(w)
	}
	return nil
}

// SerializedSize returns the serialized size of a message if it implements Serializable.
func SerializedSize(msg Message, w *binary.Writer) int {
	if s, ok := msg.(Serializable); ok {
		return s.SerializedSize(w)
	}
	return 0
}

// encodeUint encodes a variable-width unsigned integer into a buffer.
func encodeUint(buf []byte, v uint64, size int) {
	switch size {
	case 1:
		buf[0] = uint8(v)
	case 2:
		buf[0] = uint8(v)
		buf[1] = uint8(v >> 8)
	case 4:
		buf[0] = uint8(v)
		buf[1] = uint8(v >> 8)
		buf[2] = uint8(v >> 16)
		buf[3] = uint8(v >> 24)
	case 8:
		buf[0] = uint8(v)
		buf[1] = uint8(v >> 8)
		buf[2] = uint8(v >> 16)
		buf[3] = uint8(v >> 24)
		buf[4] = uint8(v >> 32)
		buf[5] = uint8(v >> 40)
		buf[6] = uint8(v >> 48)
		buf[7] = uint8(v >> 56)
	}
}
