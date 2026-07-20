package message

import (
	"encoding/binary"
	"fmt"

	binpkg "github.com/huangzhengshun/go-hdf5/internal/binary"
)

// DatatypeClass represents the class of an HDF5 datatype.
type DatatypeClass uint8

const (
	ClassFixedPoint DatatypeClass = 0  // Integers
	ClassFloatPoint DatatypeClass = 1  // Floating-point
	ClassTime       DatatypeClass = 2  // Time (rarely used)
	ClassString     DatatypeClass = 3  // Strings
	ClassBitfield   DatatypeClass = 4  // Bitfields
	ClassOpaque     DatatypeClass = 5  // Opaque data
	ClassCompound   DatatypeClass = 6  // Compound types (structs)
	ClassReference  DatatypeClass = 7  // References to objects/regions
	ClassEnum       DatatypeClass = 8  // Enumerated types
	ClassVarLen     DatatypeClass = 9  // Variable-length data
	ClassArray      DatatypeClass = 10 // Fixed-size arrays
)

// ByteOrder represents the byte order of numeric types.
type ByteOrder uint8

const (
	OrderLE     ByteOrder = 0 // Little-endian
	OrderBE     ByteOrder = 1 // Big-endian
	OrderVAX    ByteOrder = 2 // VAX mixed-endian (rare)
	OrderNone   ByteOrder = 3 // Not applicable
)

// StringPadding represents how strings are padded.
type StringPadding uint8

const (
	PadNullTerm  StringPadding = 0 // Null-terminated
	PadNullPad   StringPadding = 1 // Null-padded
	PadSpacePad  StringPadding = 2 // Space-padded
)

// CharacterSet represents the character encoding.
type CharacterSet uint8

const (
	CharsetASCII CharacterSet = 0
	CharsetUTF8  CharacterSet = 1
)

// Datatype represents a datatype message (type 0x0003).
type Datatype struct {
	Class     DatatypeClass
	ClassBits uint32 // Class-specific bit field
	Size      uint32

	// Class-specific properties
	ByteOrder ByteOrder

	// Fixed-point specific
	BitOffset    uint16
	BitPrecision uint16
	Signed       bool

	// Float-point specific (mantissa/exponent info stored in ClassBits)

	// String specific
	StringPadding StringPadding
	CharSet       CharacterSet

	// Compound specific
	Members []CompoundMember

	// Array specific
	ArrayDims []uint32
	BaseType  *Datatype

	// VarLen specific
	VarLenType    *Datatype
	IsVarLenString bool

	// Raw properties data for complex types
	Properties []byte
}

// CompoundMember represents a member of a compound datatype.
type CompoundMember struct {
	Name       string
	ByteOffset uint32
	Type       *Datatype
}

func (m *Datatype) Type() Type { return TypeDatatype }

// IsInteger returns true if this is an integer type.
func (m *Datatype) IsInteger() bool {
	return m.Class == ClassFixedPoint
}

// IsFloat returns true if this is a floating-point type.
func (m *Datatype) IsFloat() bool {
	return m.Class == ClassFloatPoint
}

// IsString returns true if this is a string type (fixed or variable-length).
func (m *Datatype) IsString() bool {
	return m.Class == ClassString || (m.Class == ClassVarLen && m.IsVarLenString)
}

// IsCompound returns true if this is a compound type.
func (m *Datatype) IsCompound() bool {
	return m.Class == ClassCompound
}

// IsArray returns true if this is an array type.
func (m *Datatype) IsArray() bool {
	return m.Class == ClassArray
}

// IsVarLen returns true if this is a variable-length type.
func (m *Datatype) IsVarLen() bool {
	return m.Class == ClassVarLen
}

func parseDatatype(data []byte, r *binpkg.Reader) (*Datatype, error) {
	dt, _, err := parseDatatypeWithSize(data, r)
	return dt, err
}

// parseDatatypeWithSize parses a datatype and returns the total bytes consumed.
func parseDatatypeWithSize(data []byte, r *binpkg.Reader) (*Datatype, int, error) {
	if len(data) < 8 {
		return nil, 0, fmt.Errorf("datatype message too short")
	}

	classAndVersion := data[0]
	class := DatatypeClass(classAndVersion & 0x0F)
	// version := classAndVersion >> 4

	classBits := uint32(data[1]) | uint32(data[2])<<8 | uint32(data[3])<<16
	size := binary.LittleEndian.Uint32(data[4:8])

	// Calculate properties size based on class
	propsSize := calcPropertiesSize(class, data[8:], classBits, size)

	dt := &Datatype{
		Class:      class,
		ClassBits:  classBits,
		Size:       size,
		Properties: data[8 : 8+propsSize],
	}

	// Parse class-specific properties
	props := data[8:]

	switch class {
	case ClassFixedPoint:
		dt.ByteOrder = ByteOrder(classBits & 0x01)
		if classBits&0x08 != 0 {
			dt.Signed = true
		}
		if len(props) >= 4 {
			dt.BitOffset = binary.LittleEndian.Uint16(props[0:2])
			dt.BitPrecision = binary.LittleEndian.Uint16(props[2:4])
		}

	case ClassFloatPoint:
		dt.ByteOrder = ByteOrder(classBits & 0x01)
		// Float properties contain bit positions for sign, exponent, mantissa
		// We store them in Properties for later use

	case ClassString:
		dt.StringPadding = StringPadding(classBits & 0x0F)
		dt.CharSet = CharacterSet((classBits >> 4) & 0x0F)

	case ClassCompound:
		numMembers := int(classBits & 0xFFFF)
		version := int(classAndVersion >> 4)
		dt.Members = make([]CompoundMember, 0, numMembers)
		offset := 0
		for i := 0; i < numMembers && offset < len(props); i++ {
			member, consumed, err := parseCompoundMember(props[offset:], r, version, size)
			if err != nil {
				break
			}
			dt.Members = append(dt.Members, member)
			offset += consumed
		}

	case ClassArray:
		if len(props) >= 1 {
			ndims := int(props[0])
			dt.ArrayDims = make([]uint32, ndims)
			offset := 4 // Skip version + reserved bytes
			for i := 0; i < ndims && offset+4 <= len(props); i++ {
				dt.ArrayDims[i] = binary.LittleEndian.Uint32(props[offset:])
				offset += 4
			}
			// Parse base type
			if offset < len(props) {
				baseType, err := parseDatatype(props[offset:], r)
				if err == nil {
					dt.BaseType = baseType
				}
			}
		}

	case ClassVarLen:
		// Type: 0 = sequence, 1 = string
		dt.IsVarLenString = (classBits & 0x0F) == 1
		if len(props) > 0 {
			varLenType, err := parseDatatype(props, r)
			if err == nil {
				dt.VarLenType = varLenType
			}
		}
	}

	return dt, 8 + propsSize, nil
}

// calcPropertiesSize calculates the size of properties for a given datatype class.
func calcPropertiesSize(class DatatypeClass, props []byte, classBits uint32, size uint32) int {
	switch class {
	case ClassFixedPoint:
		return 4 // bit offset (2) + bit precision (2)
	case ClassFloatPoint:
		return 12 // bit offset (2) + bit precision (2) + exp location (1) + exp size (1) + mant location (1) + mant size (2) + exp bias (4)
	case ClassString:
		return 0 // no properties
	case ClassBitfield:
		return 4 // bit offset (2) + bit precision (2)
	case ClassOpaque:
		// Opaque has a tag (null-terminated string)
		end := 0
		for end < len(props) && props[end] != 0 {
			end++
		}
		return end + 1 // include null
	case ClassCompound:
		// Compound properties include all member definitions
		// The size depends on member count and their types
		// For now, use all remaining data (will be trimmed by the calling context)
		return len(props)
	case ClassReference:
		return 0 // no properties
	case ClassEnum:
		// Enum has base type + name-value pairs
		// For simplicity, use remaining data
		return len(props)
	case ClassVarLen:
		// VarLen has a base type
		if len(props) >= 8 {
			baseClass := DatatypeClass(props[0] & 0x0F)
			baseProps := calcPropertiesSize(baseClass, props[8:], 0, 0)
			return 8 + baseProps
		}
		return len(props)
	case ClassArray:
		// Array has dimensions + base type
		if len(props) >= 4 {
			ndims := int(props[0])
			offset := 4 + ndims*4 // version(1) + reserved(3) + dims(4*ndims)
			if offset < len(props) {
				baseClass := DatatypeClass(props[offset] & 0x0F)
				baseProps := calcPropertiesSize(baseClass, props[offset+8:], 0, 0)
				return offset + 8 + baseProps
			}
		}
		return len(props)
	default:
		return len(props)
	}
}

func parseCompoundMember(data []byte, r *binpkg.Reader, version int, compoundSize uint32) (CompoundMember, int, error) {
	var member CompoundMember

	// Find null-terminated name
	nameEnd := 0
	for nameEnd < len(data) && data[nameEnd] != 0 {
		nameEnd++
	}
	if nameEnd >= len(data) {
		return member, 0, fmt.Errorf("compound member name not terminated")
	}

	member.Name = string(data[:nameEnd])
	offset := nameEnd + 1

	// Version 1 and 2: names are padded to 8-byte boundary
	// Version 3: no padding
	if version < 3 {
		if offset%8 != 0 {
			offset += 8 - (offset % 8)
		}
	}

	// Determine byte offset size based on compound size (version 3) or use 4 bytes (version 1/2)
	var offsetSize int
	if version >= 3 {
		// Version 3: offset size depends on compound type's total size
		if compoundSize <= 255 {
			offsetSize = 1
		} else if compoundSize <= 65535 {
			offsetSize = 2
		} else if compoundSize <= 4294967295 {
			offsetSize = 4
		} else {
			offsetSize = 8
		}
	} else {
		offsetSize = 4
	}

	if offset+offsetSize > len(data) {
		return member, 0, fmt.Errorf("compound member truncated")
	}

	switch offsetSize {
	case 1:
		member.ByteOffset = uint32(data[offset])
	case 2:
		member.ByteOffset = uint32(binary.LittleEndian.Uint16(data[offset:]))
	case 4:
		member.ByteOffset = binary.LittleEndian.Uint32(data[offset:])
	case 8:
		member.ByteOffset = uint32(binary.LittleEndian.Uint64(data[offset:]))
	}
	offset += offsetSize

	// Parse member datatype
	if offset < len(data) {
		memberType, typeSize, err := parseDatatypeWithSize(data[offset:], r)
		if err != nil {
			return member, 0, err
		}
		member.Type = memberType
		offset += typeSize
	}

	return member, offset, nil
}
