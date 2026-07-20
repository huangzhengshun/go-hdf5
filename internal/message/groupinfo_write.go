package message

import (
	"github.com/huangzhengshun/go-hdf5/internal/binary"
)

// GroupInfo represents a group info message (type 0x000A).
// This message provides information about a group's link storage.
type GroupInfo struct {
	Version           uint8
	Flags             uint8
	MaxCompactLinks   uint16 // Present if flags bit 0 set
	MinDenseLinks     uint16 // Present if flags bit 0 set
	EstNumEntries     uint16 // Present if flags bit 1 set
	EstLinkNameLen    uint16 // Present if flags bit 1 set
}

func (m *GroupInfo) Type() Type { return TypeGroupInfo }

// Serialize writes the GroupInfo to the writer.
func (m *GroupInfo) Serialize(w *binary.Writer) error {
	if err := w.WriteUint8(m.Version); err != nil {
		return err
	}

	if err := w.WriteUint8(m.Flags); err != nil {
		return err
	}

	// Link phase change values (if flags bit 0 set)
	if m.Flags&0x01 != 0 {
		if err := w.WriteUint16(m.MaxCompactLinks); err != nil {
			return err
		}
		if err := w.WriteUint16(m.MinDenseLinks); err != nil {
			return err
		}
	}

	// Estimated entries info (if flags bit 1 set)
	if m.Flags&0x02 != 0 {
		if err := w.WriteUint16(m.EstNumEntries); err != nil {
			return err
		}
		if err := w.WriteUint16(m.EstLinkNameLen); err != nil {
			return err
		}
	}

	return nil
}

// SerializedSize returns the size in bytes when serialized.
func (m *GroupInfo) SerializedSize(w *binary.Writer) int {
	// Version + flags
	size := 2

	// Link phase change values
	if m.Flags&0x01 != 0 {
		size += 4
	}

	// Estimated entries info
	if m.Flags&0x02 != 0 {
		size += 4
	}

	return size
}

// NewGroupInfo creates a new minimal GroupInfo message.
func NewGroupInfo() *GroupInfo {
	return &GroupInfo{
		Version: 0,
		Flags:   0,
	}
}
