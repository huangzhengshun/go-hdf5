package message

import (
	"github.com/huangzhengshun/go-hdf5/internal/binary"
)

// NewAttribute creates a new attribute message.
// Uses version 3 format for modern compatibility.
func NewAttribute(name string, datatype *Datatype, dataspace *Dataspace, data []byte) *Attribute {
	return &Attribute{
		Version:   3,
		Name:      name,
		Datatype:  datatype,
		Dataspace: dataspace,
		Data:      data,
	}
}

// NewScalarAttribute creates a new scalar attribute (no dimensions).
func NewScalarAttribute(name string, datatype *Datatype, data []byte) *Attribute {
	return &Attribute{
		Version:   3,
		Name:      name,
		Datatype:  datatype,
		Dataspace: NewScalarDataspace(),
		Data:      data,
	}
}

// Serialize writes the Attribute message to the writer.
func (m *Attribute) Serialize(w *binary.Writer) error {
	// Calculate sizes
	nameSize := uint16(len(m.Name) + 1) // +1 for null terminator
	datatypeSize := m.Datatype.SerializedSize(w)
	dataspaceSize := m.Dataspace.SerializedSize(w)

	// Version 3 format
	if err := w.WriteUint8(3); err != nil {
		return err
	}

	// Flags (0 for now)
	if err := w.WriteUint8(0); err != nil {
		return err
	}

	// Name size
	if err := w.WriteUint16(nameSize); err != nil {
		return err
	}

	// Datatype size
	if err := w.WriteUint16(uint16(datatypeSize)); err != nil {
		return err
	}

	// Dataspace size
	if err := w.WriteUint16(uint16(dataspaceSize)); err != nil {
		return err
	}

	// Encoding (0 = ASCII)
	if err := w.WriteUint8(0); err != nil {
		return err
	}

	// Name (null-terminated)
	if err := w.WriteBytes([]byte(m.Name)); err != nil {
		return err
	}
	if err := w.WriteUint8(0); err != nil { // null terminator
		return err
	}

	// Datatype
	if err := m.Datatype.Serialize(w); err != nil {
		return err
	}

	// Dataspace
	if err := m.Dataspace.Serialize(w); err != nil {
		return err
	}

	// Attribute data
	if err := w.WriteBytes(m.Data); err != nil {
		return err
	}

	return nil
}

// SerializedSize returns the size in bytes when serialized.
func (m *Attribute) SerializedSize(w *binary.Writer) int {
	nameSize := len(m.Name) + 1 // +1 for null terminator
	datatypeSize := m.Datatype.SerializedSize(w)
	dataspaceSize := m.Dataspace.SerializedSize(w)
	dataSize := len(m.Data)

	// Version 3: version(1) + flags(1) + nameSize(2) + dtSize(2) + dsSize(2) + encoding(1)
	// + name + datatype + dataspace + data
	return 9 + nameSize + datatypeSize + dataspaceSize + dataSize
}
