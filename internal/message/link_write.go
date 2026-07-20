package message

import (
	"github.com/huangzhengshun/go-hdf5/internal/binary"
)

// Serialize writes the Link to the writer.
// Uses version 1 format.
func (m *Link) Serialize(w *binary.Writer) error {
	// Version 1 format:
	// Byte 0: Version (1)
	// Byte 1: Flags
	//   Bits 0-1: Size of link name length (0=1, 1=2, 2=4, 3=8)
	//   Bit 2: Creation order present
	//   Bit 3: Link type present (set unless hard link)
	//   Bit 4: Link name charset present
	// Optional: Link type (1 byte if flag bit 3)
	// Optional: Creation order (8 bytes if flag bit 2)
	// Optional: Charset (1 byte if flag bit 4)
	// Link name length (1-8 bytes based on flags)
	// Link name (variable)
	// Link info (depends on type)

	if err := w.WriteUint8(1); err != nil { // Version 1
		return err
	}

	// Determine name length size
	nameLen := len(m.Name)
	var nameLenSize int
	var nameLenSizeBits uint8
	if nameLen <= 0xFF {
		nameLenSize = 1
		nameLenSizeBits = 0
	} else if nameLen <= 0xFFFF {
		nameLenSize = 2
		nameLenSizeBits = 1
	} else if nameLen <= 0xFFFFFFFF {
		nameLenSize = 4
		nameLenSizeBits = 2
	} else {
		nameLenSize = 8
		nameLenSizeBits = 3
	}

	// Build flags
	flags := nameLenSizeBits
	if m.LinkType != LinkTypeHard {
		flags |= 0x08 // Link type present
	}

	if err := w.WriteUint8(flags); err != nil {
		return err
	}

	// Write link type if not hard link
	if m.LinkType != LinkTypeHard {
		if err := w.WriteUint8(uint8(m.LinkType)); err != nil {
			return err
		}
	}

	// Write name length
	if err := w.WriteUintN(uint64(nameLen), nameLenSize); err != nil {
		return err
	}

	// Write name
	if err := w.WriteBytes([]byte(m.Name)); err != nil {
		return err
	}

	// Write link-type specific info
	switch m.LinkType {
	case LinkTypeHard:
		if err := w.WriteOffset(m.ObjectAddress); err != nil {
			return err
		}

	case LinkTypeSoft:
		softLen := uint16(len(m.SoftLinkValue))
		if err := w.WriteUint16(softLen); err != nil {
			return err
		}
		if err := w.WriteBytes([]byte(m.SoftLinkValue)); err != nil {
			return err
		}

	case LinkTypeExternal:
		// External link format: flags (1) + file (null-term) + path (null-term)
		extDataLen := 1 + len(m.ExternalFile) + 1 + len(m.ExternalPath) + 1
		if err := w.WriteUint16(uint16(extDataLen)); err != nil {
			return err
		}
		// Flags (version 0)
		if err := w.WriteUint8(0); err != nil {
			return err
		}
		// File (null-terminated)
		if err := w.WriteBytes([]byte(m.ExternalFile)); err != nil {
			return err
		}
		if err := w.WriteUint8(0); err != nil {
			return err
		}
		// Path (null-terminated)
		if err := w.WriteBytes([]byte(m.ExternalPath)); err != nil {
			return err
		}
		if err := w.WriteUint8(0); err != nil {
			return err
		}
	}

	return nil
}

// SerializedSize returns the size in bytes when serialized.
func (m *Link) SerializedSize(w *binary.Writer) int {
	// Version + flags
	size := 2

	// Link type if not hard link
	if m.LinkType != LinkTypeHard {
		size += 1
	}

	// Name length field size
	nameLen := len(m.Name)
	if nameLen <= 0xFF {
		size += 1
	} else if nameLen <= 0xFFFF {
		size += 2
	} else if nameLen <= 0xFFFFFFFF {
		size += 4
	} else {
		size += 8
	}

	// Name
	size += nameLen

	// Link-type specific info
	switch m.LinkType {
	case LinkTypeHard:
		size += w.OffsetSize()
	case LinkTypeSoft:
		size += 2 + len(m.SoftLinkValue)
	case LinkTypeExternal:
		size += 2 + 1 + len(m.ExternalFile) + 1 + len(m.ExternalPath) + 1
	}

	return size
}

// NewHardLink creates a new hard link message.
func NewHardLink(name string, objectAddress uint64) *Link {
	return &Link{
		Version:       1,
		LinkType:      LinkTypeHard,
		Name:          name,
		ObjectAddress: objectAddress,
	}
}

// NewSoftLink creates a new soft link message.
func NewSoftLink(name string, targetPath string) *Link {
	return &Link{
		Version:       1,
		LinkType:      LinkTypeSoft,
		Name:          name,
		SoftLinkValue: targetPath,
	}
}

// NewExternalLink creates a new external link message.
func NewExternalLink(name string, externalFile, externalPath string) *Link {
	return &Link{
		Version:      1,
		LinkType:     LinkTypeExternal,
		Name:         name,
		ExternalFile: externalFile,
		ExternalPath: externalPath,
	}
}
