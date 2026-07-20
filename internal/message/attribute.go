package message

import (
	"encoding/binary"
	"fmt"

	binpkg "github.com/huangzhengshun/go-hdf5/internal/binary"
)

// Attribute represents an attribute message (type 0x000C).
type Attribute struct {
	Version      uint8
	Name         string
	DatatypeSize uint16
	DataspaceSize uint16
	Datatype     *Datatype
	Dataspace    *Dataspace
	Data         []byte
}

func (m *Attribute) Type() Type { return TypeAttribute }

func parseAttribute(data []byte, r *binpkg.Reader) (*Attribute, error) {
	if len(data) < 6 {
		return nil, fmt.Errorf("attribute message too short")
	}

	attr := &Attribute{
		Version: data[0],
	}

	switch attr.Version {
	case 1:
		return parseAttributeV1(data, r, attr)
	case 2:
		return parseAttributeV2(data, r, attr)
	case 3:
		return parseAttributeV3(data, r, attr)
	default:
		return nil, fmt.Errorf("unsupported attribute version: %d", attr.Version)
	}
}

func parseAttributeV1(data []byte, r *binpkg.Reader, attr *Attribute) (*Attribute, error) {
	if len(data) < 8 {
		return nil, fmt.Errorf("attribute v1 too short")
	}

	nameSize := binary.LittleEndian.Uint16(data[2:4])
	attr.DatatypeSize = binary.LittleEndian.Uint16(data[4:6])
	attr.DataspaceSize = binary.LittleEndian.Uint16(data[6:8])

	offset := 8

	// Parse name (null-padded to 8-byte boundary)
	if offset+int(nameSize) > len(data) {
		return nil, fmt.Errorf("attribute name truncated")
	}
	nameEnd := offset
	for nameEnd < offset+int(nameSize) && data[nameEnd] != 0 {
		nameEnd++
	}
	attr.Name = string(data[offset:nameEnd])
	offset += int(nameSize)

	// Pad to 8-byte boundary
	if offset%8 != 0 {
		offset += 8 - (offset % 8)
	}

	// Parse datatype
	if offset+int(attr.DatatypeSize) > len(data) {
		return nil, fmt.Errorf("attribute datatype truncated")
	}
	dt, err := parseDatatype(data[offset:offset+int(attr.DatatypeSize)], r)
	if err == nil {
		attr.Datatype = dt
	}
	offset += int(attr.DatatypeSize)

	// Pad to 8-byte boundary
	if offset%8 != 0 {
		offset += 8 - (offset % 8)
	}

	// Parse dataspace
	if offset+int(attr.DataspaceSize) > len(data) {
		return nil, fmt.Errorf("attribute dataspace truncated")
	}
	ds, err := parseDataspace(data[offset:offset+int(attr.DataspaceSize)], r)
	if err == nil {
		attr.Dataspace = ds
	}
	offset += int(attr.DataspaceSize)

	// Pad to 8-byte boundary
	if offset%8 != 0 {
		offset += 8 - (offset % 8)
	}

	// Remaining data is the attribute value
	if offset < len(data) {
		attr.Data = make([]byte, len(data)-offset)
		copy(attr.Data, data[offset:])
	}

	return attr, nil
}

func parseAttributeV2(data []byte, r *binpkg.Reader, attr *Attribute) (*Attribute, error) {
	if len(data) < 6 {
		return nil, fmt.Errorf("attribute v2 too short")
	}

	// flags := data[1]
	nameSize := binary.LittleEndian.Uint16(data[2:4])
	attr.DatatypeSize = binary.LittleEndian.Uint16(data[4:6])
	attr.DataspaceSize = binary.LittleEndian.Uint16(data[6:8])

	offset := 8

	// Parse name (NOT padded in v2)
	if offset+int(nameSize) > len(data) {
		return nil, fmt.Errorf("attribute name truncated")
	}
	nameEnd := offset
	for nameEnd < offset+int(nameSize) && data[nameEnd] != 0 {
		nameEnd++
	}
	attr.Name = string(data[offset:nameEnd])
	offset += int(nameSize)

	// Parse datatype
	if offset+int(attr.DatatypeSize) > len(data) {
		return nil, fmt.Errorf("attribute datatype truncated")
	}
	dt, err := parseDatatype(data[offset:offset+int(attr.DatatypeSize)], r)
	if err == nil {
		attr.Datatype = dt
	}
	offset += int(attr.DatatypeSize)

	// Parse dataspace
	if offset+int(attr.DataspaceSize) > len(data) {
		return nil, fmt.Errorf("attribute dataspace truncated")
	}
	ds, err := parseDataspace(data[offset:offset+int(attr.DataspaceSize)], r)
	if err == nil {
		attr.Dataspace = ds
	}
	offset += int(attr.DataspaceSize)

	// Remaining data is the attribute value
	if offset < len(data) {
		attr.Data = make([]byte, len(data)-offset)
		copy(attr.Data, data[offset:])
	}

	return attr, nil
}

func parseAttributeV3(data []byte, r *binpkg.Reader, attr *Attribute) (*Attribute, error) {
	// V3 is similar to V2 but with encoding field
	if len(data) < 9 {
		return nil, fmt.Errorf("attribute v3 too short")
	}

	// flags := data[1]
	nameSize := binary.LittleEndian.Uint16(data[2:4])
	attr.DatatypeSize = binary.LittleEndian.Uint16(data[4:6])
	attr.DataspaceSize = binary.LittleEndian.Uint16(data[6:8])
	// encoding := data[8]

	offset := 9

	// Parse name
	if offset+int(nameSize) > len(data) {
		return nil, fmt.Errorf("attribute name truncated")
	}
	nameEnd := offset
	for nameEnd < offset+int(nameSize) && data[nameEnd] != 0 {
		nameEnd++
	}
	attr.Name = string(data[offset:nameEnd])
	offset += int(nameSize)

	// Parse datatype
	if offset+int(attr.DatatypeSize) > len(data) {
		return nil, fmt.Errorf("attribute datatype truncated")
	}
	dt, err := parseDatatype(data[offset:offset+int(attr.DatatypeSize)], r)
	if err == nil {
		attr.Datatype = dt
	}
	offset += int(attr.DatatypeSize)

	// Parse dataspace
	if offset+int(attr.DataspaceSize) > len(data) {
		return nil, fmt.Errorf("attribute dataspace truncated")
	}
	ds, err := parseDataspace(data[offset:offset+int(attr.DataspaceSize)], r)
	if err == nil {
		attr.Dataspace = ds
	}
	offset += int(attr.DataspaceSize)

	// Remaining data is the attribute value
	if offset < len(data) {
		attr.Data = make([]byte, len(data)-offset)
		copy(attr.Data, data[offset:])
	}

	return attr, nil
}
