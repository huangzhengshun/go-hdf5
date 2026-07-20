// Package message handles parsing of HDF5 header messages.
//
// Header messages are embedded in object headers and contain metadata
// about dataspace, datatype, storage layout, filters, attributes, etc.
package message

import (
	"fmt"

	"github.com/huangzhengshun/go-hdf5/internal/binary"
)

// Type represents an HDF5 header message type.
type Type uint16

// Header message types
const (
	TypeNIL                      Type = 0x0000
	TypeDataspace                Type = 0x0001
	TypeLinkInfo                 Type = 0x0002
	TypeDatatype                 Type = 0x0003
	TypeFillValueOld             Type = 0x0004
	TypeFillValue                Type = 0x0005
	TypeLink                     Type = 0x0006
	TypeExternalDataFiles        Type = 0x0007
	TypeDataLayout               Type = 0x0008
	TypeBogus                    Type = 0x0009
	TypeGroupInfo                Type = 0x000A
	TypeFilterPipeline           Type = 0x000B
	TypeAttribute                Type = 0x000C
	TypeObjectComment            Type = 0x000D
	TypeObjectModTime            Type = 0x000E
	TypeSharedMessageTable       Type = 0x000F
	TypeObjectHeaderContinuation Type = 0x0010
	TypeSymbolTable              Type = 0x0011
	TypeObjectModTimeOld         Type = 0x0012
	TypeBTreeKValues             Type = 0x0013
	TypeDriverInfo               Type = 0x0014
	TypeAttributeInfo            Type = 0x0015
	TypeObjectRefCount           Type = 0x0016
)

// Message is the interface implemented by all header messages.
type Message interface {
	Type() Type
}

// Parse parses a header message from raw bytes.
func Parse(typ Type, data []byte, flags uint8, r *binary.Reader) (Message, error) {
	switch typ {
	case TypeDataspace:
		return parseDataspace(data, r)
	case TypeDatatype:
		return parseDatatype(data, r)
	case TypeDataLayout:
		return parseDataLayout(data, r)
	case TypeFilterPipeline:
		return parseFilterPipeline(data, r)
	case TypeFillValue:
		return parseFillValue(data, r)
	case TypeAttribute:
		return parseAttribute(data, r)
	case TypeLink:
		return parseLink(data, r)
	case TypeSymbolTable:
		return parseSymbolTable(data, r)
	case TypeObjectHeaderContinuation:
		return ParseContinuation(data, r)
	default:
		// Return an unknown message wrapper for unhandled types
		return &Unknown{typ: typ, data: data}, nil
	}
}

// Unknown represents an unrecognized message type.
type Unknown struct {
	typ  Type
	data []byte
}

func (m *Unknown) Type() Type    { return m.typ }
func (m *Unknown) Data() []byte  { return m.data }

// Continuation represents an object header continuation message.
type Continuation struct {
	Offset uint64
	Length uint64
}

func (m *Continuation) Type() Type { return TypeObjectHeaderContinuation }

// ParseContinuation parses a continuation message.
func ParseContinuation(data []byte, r *binary.Reader) (*Continuation, error) {
	if len(data) < 2*r.OffsetSize() {
		return nil, fmt.Errorf("continuation message too short")
	}

	offsetSize := r.OffsetSize()

	offset := decodeUint(data[0:offsetSize], offsetSize, r.ByteOrder())
	length := decodeUint(data[offsetSize:2*offsetSize], offsetSize, r.ByteOrder())

	return &Continuation{
		Offset: offset,
		Length: length,
	}, nil
}
