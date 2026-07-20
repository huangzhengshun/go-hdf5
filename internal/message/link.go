package message

import (
	"encoding/binary"
	"fmt"

	binpkg "github.com/huangzhengshun/go-hdf5/internal/binary"
)

// LinkType represents the type of link.
type LinkType uint8

const (
	LinkTypeHard     LinkType = 0  // Hard link (object header address)
	LinkTypeSoft     LinkType = 1  // Soft link (path string)
	LinkTypeExternal LinkType = 64 // External link (file + path) - per HDF5 spec
)

// Link represents a link message (type 0x0006).
type Link struct {
	Version      uint8
	LinkType     LinkType
	CreationOrder uint64
	Name         string
	Charset      uint8

	// Hard link
	ObjectAddress uint64

	// Soft link
	SoftLinkValue string

	// External link
	ExternalFile string
	ExternalPath string
}

func (m *Link) Type() Type { return TypeLink }

// IsHard returns true if this is a hard link.
func (m *Link) IsHard() bool {
	return m.LinkType == LinkTypeHard
}

// IsSoft returns true if this is a soft link.
func (m *Link) IsSoft() bool {
	return m.LinkType == LinkTypeSoft
}

// IsExternal returns true if this is an external link.
func (m *Link) IsExternal() bool {
	return m.LinkType == LinkTypeExternal
}

func parseLink(data []byte, r *binpkg.Reader) (*Link, error) {
	if len(data) < 2 {
		return nil, fmt.Errorf("link message too short")
	}

	link := &Link{
		Version: data[0],
	}

	flags := data[1]
	offset := 2

	// Size of link name length field
	nameLenSize := 1 << (flags & 0x03)

	// Link type present (flag bit 3)
	if flags&0x08 != 0 {
		if offset >= len(data) {
			return nil, fmt.Errorf("link type truncated")
		}
		link.LinkType = LinkType(data[offset])
		offset++
	}

	// Creation order present (flag bit 2)
	if flags&0x04 != 0 {
		if offset+8 > len(data) {
			return nil, fmt.Errorf("link creation order truncated")
		}
		link.CreationOrder = binary.LittleEndian.Uint64(data[offset:])
		offset += 8
	}

	// Link name charset (flag bit 4)
	if flags&0x10 != 0 {
		if offset >= len(data) {
			return nil, fmt.Errorf("link charset truncated")
		}
		link.Charset = data[offset]
		offset++
	}

	// Parse link name length
	if offset+nameLenSize > len(data) {
		return nil, fmt.Errorf("link name length truncated")
	}
	var nameLen uint64
	switch nameLenSize {
	case 1:
		nameLen = uint64(data[offset])
	case 2:
		nameLen = uint64(binary.LittleEndian.Uint16(data[offset:]))
	case 4:
		nameLen = uint64(binary.LittleEndian.Uint32(data[offset:]))
	case 8:
		nameLen = binary.LittleEndian.Uint64(data[offset:])
	}
	offset += nameLenSize

	// Parse link name
	if offset+int(nameLen) > len(data) {
		return nil, fmt.Errorf("link name truncated")
	}
	link.Name = string(data[offset : offset+int(nameLen)])
	offset += int(nameLen)

	// Parse link info based on type
	switch link.LinkType {
	case LinkTypeHard:
		offsetSize := r.OffsetSize()
		if offset+offsetSize > len(data) {
			return nil, fmt.Errorf("hard link address truncated")
		}
		link.ObjectAddress = decodeUint(data[offset:], offsetSize, r.ByteOrder())

	case LinkTypeSoft:
		if offset+2 > len(data) {
			return nil, fmt.Errorf("soft link length truncated")
		}
		softLen := binary.LittleEndian.Uint16(data[offset:])
		offset += 2
		if offset+int(softLen) > len(data) {
			return nil, fmt.Errorf("soft link value truncated")
		}
		link.SoftLinkValue = string(data[offset : offset+int(softLen)])

	case LinkTypeExternal:
		if offset+2 > len(data) {
			return nil, fmt.Errorf("external link length truncated")
		}
		extLen := binary.LittleEndian.Uint16(data[offset:])
		offset += 2
		if offset+int(extLen) > len(data) {
			return nil, fmt.Errorf("external link value truncated")
		}
		// External link format: flags (1) + file (null-term) + path (null-term)
		extData := data[offset : offset+int(extLen)]
		if len(extData) < 2 {
			return nil, fmt.Errorf("external link data too short")
		}
		// Skip flags byte
		extData = extData[1:]
		// Find file name
		fileEnd := 0
		for fileEnd < len(extData) && extData[fileEnd] != 0 {
			fileEnd++
		}
		link.ExternalFile = string(extData[:fileEnd])
		if fileEnd+1 < len(extData) {
			link.ExternalPath = string(extData[fileEnd+1:])
			// Remove trailing null if present
			if len(link.ExternalPath) > 0 && link.ExternalPath[len(link.ExternalPath)-1] == 0 {
				link.ExternalPath = link.ExternalPath[:len(link.ExternalPath)-1]
			}
		}
	}

	return link, nil
}
