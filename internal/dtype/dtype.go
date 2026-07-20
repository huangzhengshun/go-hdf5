// Package dtype provides datatype handling and conversion for HDF5 data.
//
// This package works with the message.Datatype parsed from object headers
// and provides utilities for converting raw HDF5 data to Go types.
package dtype

import (
	"encoding/binary"
	"fmt"
	"reflect"

	"github.com/huangzhengshun/go-hdf5/internal/message"
)

// GoType returns the Go reflect.Type that corresponds to the given HDF5 datatype.
func GoType(dt *message.Datatype) (reflect.Type, error) {
	if dt == nil {
		return nil, fmt.Errorf("nil datatype")
	}

	switch dt.Class {
	case message.ClassFixedPoint:
		return goTypeFixedPoint(dt)
	case message.ClassFloatPoint:
		return goTypeFloatPoint(dt)
	case message.ClassString:
		return reflect.TypeOf(""), nil
	case message.ClassCompound:
		return goTypeCompound(dt)
	case message.ClassArray:
		return goTypeArray(dt)
	case message.ClassVarLen:
		if dt.IsVarLenString {
			return reflect.TypeOf(""), nil
		}
		if dt.VarLenType != nil {
			elemType, err := GoType(dt.VarLenType)
			if err != nil {
				return nil, err
			}
			return reflect.SliceOf(elemType), nil
		}
		return reflect.TypeOf([]byte{}), nil
	case message.ClassEnum:
		// Enums are stored as their base type (usually integer)
		return goTypeFixedPoint(dt)
	default:
		return nil, fmt.Errorf("unsupported datatype class: %d", dt.Class)
	}
}

func goTypeFixedPoint(dt *message.Datatype) (reflect.Type, error) {
	signed := dt.Signed

	switch dt.Size {
	case 1:
		if signed {
			return reflect.TypeOf(int8(0)), nil
		}
		return reflect.TypeOf(uint8(0)), nil
	case 2:
		if signed {
			return reflect.TypeOf(int16(0)), nil
		}
		return reflect.TypeOf(uint16(0)), nil
	case 4:
		if signed {
			return reflect.TypeOf(int32(0)), nil
		}
		return reflect.TypeOf(uint32(0)), nil
	case 8:
		if signed {
			return reflect.TypeOf(int64(0)), nil
		}
		return reflect.TypeOf(uint64(0)), nil
	default:
		return nil, fmt.Errorf("unsupported fixed-point size: %d", dt.Size)
	}
}

func goTypeFloatPoint(dt *message.Datatype) (reflect.Type, error) {
	switch dt.Size {
	case 4:
		return reflect.TypeOf(float32(0)), nil
	case 8:
		return reflect.TypeOf(float64(0)), nil
	default:
		return nil, fmt.Errorf("unsupported float size: %d", dt.Size)
	}
}

func goTypeCompound(dt *message.Datatype) (reflect.Type, error) {
	if len(dt.Members) == 0 {
		return nil, fmt.Errorf("compound type has no members")
	}

	fields := make([]reflect.StructField, len(dt.Members))
	for i, member := range dt.Members {
		memberType, err := GoType(member.Type)
		if err != nil {
			return nil, fmt.Errorf("compound member %q: %w", member.Name, err)
		}
		fields[i] = reflect.StructField{
			Name: exportName(member.Name),
			Type: memberType,
		}
	}

	return reflect.StructOf(fields), nil
}

func goTypeArray(dt *message.Datatype) (reflect.Type, error) {
	if dt.BaseType == nil {
		return nil, fmt.Errorf("array type has no base type")
	}
	if len(dt.ArrayDims) == 0 {
		return nil, fmt.Errorf("array type has no dimensions")
	}

	elemType, err := GoType(dt.BaseType)
	if err != nil {
		return nil, err
	}

	// Build nested array type from innermost to outermost
	result := elemType
	for i := len(dt.ArrayDims) - 1; i >= 0; i-- {
		result = reflect.ArrayOf(int(dt.ArrayDims[i]), result)
	}

	return result, nil
}

// exportName converts an HDF5 member name to a valid exported Go field name.
func exportName(name string) string {
	if len(name) == 0 {
		return "Field"
	}

	runes := []rune(name)

	// Capitalize first letter
	if runes[0] >= 'a' && runes[0] <= 'z' {
		runes[0] = runes[0] - 'a' + 'A'
	}

	// Replace invalid characters with underscores
	for i := range runes {
		r := runes[i]
		if !((r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') ||
			(r >= '0' && r <= '9') || r == '_') {
			runes[i] = '_'
		}
	}

	return string(runes)
}

// ByteOrder returns the binary.ByteOrder for the datatype.
func ByteOrder(dt *message.Datatype) binary.ByteOrder {
	if dt.ByteOrder == message.OrderBE {
		return binary.BigEndian
	}
	return binary.LittleEndian
}

// ElementSize returns the size of a single element in bytes.
func ElementSize(dt *message.Datatype) int {
	return int(dt.Size)
}

// IsNumeric returns true if the datatype is a numeric type.
func IsNumeric(dt *message.Datatype) bool {
	return dt.Class == message.ClassFixedPoint || dt.Class == message.ClassFloatPoint
}
