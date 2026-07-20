package message

import (
	"github.com/huangzhengshun/go-hdf5/internal/binary"
)

// Serialize writes the Datatype to the writer.
func (m *Datatype) Serialize(w *binary.Writer) error {
	// Datatype message format:
	// Byte 0: Class (lower 4 bits) + Version (upper 4 bits)
	// Bytes 1-3: Class-specific bit fields (24 bits)
	// Bytes 4-7: Size (32 bits)
	// Bytes 8+: Class-specific properties

	// Use version 1 for most types, version 3 for compound
	version := uint8(1)
	if m.Class == ClassCompound {
		version = 3
	}

	classAndVersion := uint8(m.Class) | (version << 4)
	if err := w.WriteUint8(classAndVersion); err != nil {
		return err
	}

	// Write class bits (3 bytes, little-endian)
	if err := w.WriteUint8(uint8(m.ClassBits)); err != nil {
		return err
	}
	if err := w.WriteUint8(uint8(m.ClassBits >> 8)); err != nil {
		return err
	}
	if err := w.WriteUint8(uint8(m.ClassBits >> 16)); err != nil {
		return err
	}

	// Write size
	if err := w.WriteUint32(m.Size); err != nil {
		return err
	}

	// Write class-specific properties
	switch m.Class {
	case ClassFixedPoint:
		// Bit offset (2 bytes)
		if err := w.WriteUint16(m.BitOffset); err != nil {
			return err
		}
		// Bit precision (2 bytes)
		if err := w.WriteUint16(m.BitPrecision); err != nil {
			return err
		}

	case ClassFloatPoint:
		// Float properties (12 bytes)
		if len(m.Properties) >= 12 {
			if err := w.WriteBytes(m.Properties[:12]); err != nil {
				return err
			}
		} else {
			// Write standard IEEE float properties
			if err := writeStandardFloatProperties(w, m.Size); err != nil {
				return err
			}
		}

	case ClassString:
		// No properties for strings

	case ClassCompound:
		// Write member definitions
		for _, member := range m.Members {
			if err := writeCompoundMember(w, &member, m.Size); err != nil {
				return err
			}
		}

	case ClassArray:
		// Version
		if err := w.WriteUint8(2); err != nil {
			return err
		}
		// Number of dimensions
		if err := w.WriteUint8(uint8(len(m.ArrayDims))); err != nil {
			return err
		}
		// Reserved (2 bytes)
		if err := w.WriteUint16(0); err != nil {
			return err
		}
		// Dimensions
		for _, dim := range m.ArrayDims {
			if err := w.WriteUint32(dim); err != nil {
				return err
			}
		}
		// Base type
		if m.BaseType != nil {
			if err := m.BaseType.Serialize(w); err != nil {
				return err
			}
		}

	case ClassVarLen:
		// Base type
		if m.VarLenType != nil {
			if err := m.VarLenType.Serialize(w); err != nil {
				return err
			}
		}
	}

	return nil
}

// SerializedSize returns the size in bytes when serialized.
func (m *Datatype) SerializedSize(w *binary.Writer) int {
	// Header: 8 bytes (class+version, class bits, size)
	size := 8

	switch m.Class {
	case ClassFixedPoint:
		size += 4 // bit offset + bit precision
	case ClassFloatPoint:
		size += 12 // float properties
	case ClassString:
		// no properties
	case ClassCompound:
		for _, member := range m.Members {
			size += compoundMemberSize(&member, m.Size)
		}
	case ClassArray:
		size += 4 + len(m.ArrayDims)*4 // version + ndims + reserved + dims
		if m.BaseType != nil {
			size += m.BaseType.SerializedSize(w)
		}
	case ClassVarLen:
		if m.VarLenType != nil {
			size += m.VarLenType.SerializedSize(w)
		}
	}

	return size
}

// writeStandardFloatProperties writes IEEE 754 float properties (12 bytes total).
// Format: bit_offset(2) + bit_precision(2) + exp_loc(1) + exp_size(1) + mant_loc(1) + mant_size(1) + exp_bias(4)
func writeStandardFloatProperties(w *binary.Writer, size uint32) error {
	switch size {
	case 4: // IEEE 754 single precision
		// Bit offset: 0
		if err := w.WriteUint16(0); err != nil {
			return err
		}
		// Bit precision: 32
		if err := w.WriteUint16(32); err != nil {
			return err
		}
		// Exponent location: 23
		if err := w.WriteUint8(23); err != nil {
			return err
		}
		// Exponent size: 8
		if err := w.WriteUint8(8); err != nil {
			return err
		}
		// Mantissa location: 0
		if err := w.WriteUint8(0); err != nil {
			return err
		}
		// Mantissa size: 23 (1 byte, not 2!)
		if err := w.WriteUint8(23); err != nil {
			return err
		}
		// Exponent bias: 127
		if err := w.WriteUint32(127); err != nil {
			return err
		}
	case 8: // IEEE 754 double precision
		// Bit offset: 0
		if err := w.WriteUint16(0); err != nil {
			return err
		}
		// Bit precision: 64
		if err := w.WriteUint16(64); err != nil {
			return err
		}
		// Exponent location: 52
		if err := w.WriteUint8(52); err != nil {
			return err
		}
		// Exponent size: 11
		if err := w.WriteUint8(11); err != nil {
			return err
		}
		// Mantissa location: 0
		if err := w.WriteUint8(0); err != nil {
			return err
		}
		// Mantissa size: 52 (1 byte, not 2!)
		if err := w.WriteUint8(52); err != nil {
			return err
		}
		// Exponent bias: 1023
		if err := w.WriteUint32(1023); err != nil {
			return err
		}
	default:
		// Unknown float size, write zeros
		return w.WriteZeros(12)
	}
	return nil
}

// writeCompoundMember writes a compound member definition.
func writeCompoundMember(w *binary.Writer, member *CompoundMember, compoundSize uint32) error {
	// Version 3 format: no padding, variable offset size

	// Name (null-terminated)
	if err := w.WriteBytes([]byte(member.Name)); err != nil {
		return err
	}
	if err := w.WriteUint8(0); err != nil {
		return err
	}

	// Byte offset (size depends on compound total size)
	offsetSize := memberOffsetSize(compoundSize)
	if err := w.WriteUintN(uint64(member.ByteOffset), offsetSize); err != nil {
		return err
	}

	// Member type
	if member.Type != nil {
		if err := member.Type.Serialize(w); err != nil {
			return err
		}
	}

	return nil
}

// compoundMemberSize calculates the serialized size of a compound member.
func compoundMemberSize(member *CompoundMember, compoundSize uint32) int {
	size := len(member.Name) + 1 // name + null
	size += memberOffsetSize(compoundSize)
	if member.Type != nil {
		size += member.Type.SerializedSize(nil)
	}
	return size
}

// memberOffsetSize returns the size in bytes for member offset based on compound size.
func memberOffsetSize(compoundSize uint32) int {
	if compoundSize <= 255 {
		return 1
	} else if compoundSize <= 65535 {
		return 2
	}
	return 4
}

// NewFixedPointDatatype creates a new fixed-point (integer) datatype.
func NewFixedPointDatatype(size uint32, signed bool, byteOrder ByteOrder) *Datatype {
	classBits := uint32(byteOrder)
	if signed {
		classBits |= 0x08 // Signed flag
	}

	return &Datatype{
		Class:        ClassFixedPoint,
		ClassBits:    classBits,
		Size:         size,
		ByteOrder:    byteOrder,
		BitOffset:    0,
		BitPrecision: uint16(size * 8),
		Signed:       signed,
	}
}

// NewFloatDatatype creates a new floating-point datatype.
func NewFloatDatatype(size uint32, byteOrder ByteOrder) *Datatype {
	// ClassBits for floating-point (matches h5py encoding):
	// Byte 0 (bits 0-7):
	//   - Bit 0: Byte order (0=LE, 1=BE)
	//   - Bit 5: Mantissa normalization (1=always set MSB)
	//   - Bits 6-7: Padding type
	// Byte 1 (bits 8-15): Sign location (bit position of sign bit)
	// Byte 2 (bits 16-23): Reserved (0)

	var signLocation uint32
	var props []byte

	switch size {
	case 4: // IEEE 754 single precision
		signLocation = 31 // Sign bit at bit 31
		props = []byte{
			0, 0, // bit offset (2 bytes)
			32, 0, // bit precision (2 bytes)
			23,   // exp location (1 byte)
			8,    // exp size (1 byte)
			0,    // mant location (1 byte)
			23,   // mant size (1 byte, NOT 2!)
			127, 0, 0, 0, // exp bias (4 bytes)
		}
	case 8: // IEEE 754 double precision
		signLocation = 63 // Sign bit at bit 63
		props = []byte{
			0, 0, // bit offset (2 bytes)
			64, 0, // bit precision (2 bytes)
			52,   // exp location (1 byte)
			11,   // exp size (1 byte)
			0,    // mant location (1 byte)
			52,   // mant size (1 byte, NOT 2!)
			255, 3, 0, 0, // exp bias = 1023 (4 bytes)
		}
	}

	// Build class bits: byte order (bit 0) + mantissa norm (bit 5) + sign location (byte 1)
	// Mantissa normalization = 1 means "always set MSB" (IEEE 754 normalized)
	classBits := uint32(byteOrder) | (1 << 5) | (signLocation << 8)

	return &Datatype{
		Class:      ClassFloatPoint,
		ClassBits:  classBits,
		Size:       size,
		ByteOrder:  byteOrder,
		Properties: props,
	}
}

// NewStringDatatype creates a new fixed-length string datatype.
func NewStringDatatype(size uint32, padding StringPadding, charset CharacterSet) *Datatype {
	classBits := uint32(padding) | (uint32(charset) << 4)

	return &Datatype{
		Class:         ClassString,
		ClassBits:     classBits,
		Size:          size,
		StringPadding: padding,
		CharSet:       charset,
	}
}

// NewVarLenStringDatatype creates a new variable-length string datatype.
func NewVarLenStringDatatype(charset CharacterSet) *Datatype {
	// VarLen string: type=1 (string), padding=nullterm, charset
	classBits := uint32(1) | (uint32(charset) << 4)

	// Base type for var-len string is a 1-byte fixed string
	baseType := &Datatype{
		Class:         ClassString,
		ClassBits:     uint32(PadNullTerm) | (uint32(charset) << 4),
		Size:          1,
		StringPadding: PadNullTerm,
		CharSet:       charset,
	}

	return &Datatype{
		Class:          ClassVarLen,
		ClassBits:      classBits,
		Size:           16, // hvl_t structure size (typically 16 bytes)
		VarLenType:     baseType,
		IsVarLenString: true,
	}
}

// NewCompoundDatatype creates a new compound datatype.
func NewCompoundDatatype(size uint32, members []CompoundMember) *Datatype {
	return &Datatype{
		Class:     ClassCompound,
		ClassBits: uint32(len(members)),
		Size:      size,
		Members:   members,
	}
}

// NewArrayDatatype creates a new array datatype.
func NewArrayDatatype(dims []uint32, baseType *Datatype) *Datatype {
	// Calculate total size
	totalElements := uint32(1)
	for _, d := range dims {
		totalElements *= d
	}
	size := totalElements * baseType.Size

	return &Datatype{
		Class:     ClassArray,
		Size:      size,
		ArrayDims: dims,
		BaseType:  baseType,
	}
}
