package dtype

import (
	"encoding/binary"
	"fmt"
	"math"
	"reflect"

	"github.com/huangzhengshun/go-hdf5/internal/message"
)

// Encode converts Go values to raw HDF5 bytes.
// The src parameter should be a slice or array of the appropriate type.
func Encode(dt *message.Datatype, src interface{}) ([]byte, error) {
	if dt == nil {
		return nil, fmt.Errorf("nil datatype")
	}

	srcVal := reflect.ValueOf(src)

	// Handle pointer to slice/array
	if srcVal.Kind() == reflect.Ptr {
		srcVal = srcVal.Elem()
	}

	switch dt.Class {
	case message.ClassFixedPoint:
		return encodeFixedPoint(dt, srcVal)
	case message.ClassFloatPoint:
		return encodeFloatPoint(dt, srcVal)
	case message.ClassString:
		return encodeString(dt, srcVal)
	case message.ClassVarLen:
		return encodeVarLen(dt, srcVal)
	default:
		return nil, fmt.Errorf("unsupported datatype class for encoding: %d", dt.Class)
	}
}

// EncodeScalar encodes a single scalar value.
func EncodeScalar(dt *message.Datatype, src interface{}) ([]byte, error) {
	// Wrap scalar in slice for encoding
	srcVal := reflect.ValueOf(src)
	sliceVal := reflect.MakeSlice(reflect.SliceOf(srcVal.Type()), 1, 1)
	sliceVal.Index(0).Set(srcVal)
	return Encode(dt, sliceVal.Interface())
}

func encodeFixedPoint(dt *message.Datatype, srcVal reflect.Value) ([]byte, error) {
	var order binary.ByteOrder = binary.LittleEndian
	if dt.ByteOrder == message.OrderBE {
		order = binary.BigEndian
	}

	size := int(dt.Size)
	var n int

	switch srcVal.Kind() {
	case reflect.Slice, reflect.Array:
		n = srcVal.Len()
	default:
		// Scalar value
		n = 1
		sliceVal := reflect.MakeSlice(reflect.SliceOf(srcVal.Type()), 1, 1)
		sliceVal.Index(0).Set(srcVal)
		srcVal = sliceVal
	}

	data := make([]byte, n*size)

	for i := 0; i < n; i++ {
		elem := srcVal.Index(i)
		offset := i * size

		switch elem.Kind() {
		case reflect.Int8:
			data[offset] = byte(elem.Int())
		case reflect.Int16:
			order.PutUint16(data[offset:], uint16(elem.Int()))
		case reflect.Int32:
			order.PutUint32(data[offset:], uint32(elem.Int()))
		case reflect.Int64, reflect.Int:
			order.PutUint64(data[offset:], uint64(elem.Int()))
		case reflect.Uint8:
			data[offset] = byte(elem.Uint())
		case reflect.Uint16:
			order.PutUint16(data[offset:], uint16(elem.Uint()))
		case reflect.Uint32:
			order.PutUint32(data[offset:], uint32(elem.Uint()))
		case reflect.Uint64, reflect.Uint:
			order.PutUint64(data[offset:], elem.Uint())
		default:
			return nil, fmt.Errorf("cannot encode %v as fixed-point", elem.Kind())
		}
	}

	return data, nil
}

func encodeFloatPoint(dt *message.Datatype, srcVal reflect.Value) ([]byte, error) {
	var order binary.ByteOrder = binary.LittleEndian
	if dt.ByteOrder == message.OrderBE {
		order = binary.BigEndian
	}

	size := int(dt.Size)
	var n int

	switch srcVal.Kind() {
	case reflect.Slice, reflect.Array:
		n = srcVal.Len()
	default:
		n = 1
		sliceVal := reflect.MakeSlice(reflect.SliceOf(srcVal.Type()), 1, 1)
		sliceVal.Index(0).Set(srcVal)
		srcVal = sliceVal
	}

	data := make([]byte, n*size)

	for i := 0; i < n; i++ {
		elem := srcVal.Index(i)
		offset := i * size

		switch elem.Kind() {
		case reflect.Float32:
			if size == 4 {
				order.PutUint32(data[offset:], math.Float32bits(float32(elem.Float())))
			} else {
				order.PutUint64(data[offset:], math.Float64bits(elem.Float()))
			}
		case reflect.Float64:
			if size == 4 {
				order.PutUint32(data[offset:], math.Float32bits(float32(elem.Float())))
			} else {
				order.PutUint64(data[offset:], math.Float64bits(elem.Float()))
			}
		default:
			return nil, fmt.Errorf("cannot encode %v as float", elem.Kind())
		}
	}

	return data, nil
}

func encodeString(dt *message.Datatype, srcVal reflect.Value) ([]byte, error) {
	size := int(dt.Size)
	var n int

	switch srcVal.Kind() {
	case reflect.Slice, reflect.Array:
		n = srcVal.Len()
	case reflect.String:
		// Single string
		n = 1
		sliceVal := reflect.MakeSlice(reflect.SliceOf(srcVal.Type()), 1, 1)
		sliceVal.Index(0).Set(srcVal)
		srcVal = sliceVal
	default:
		return nil, fmt.Errorf("cannot encode %v as string", srcVal.Kind())
	}

	data := make([]byte, n*size)

	for i := 0; i < n; i++ {
		elem := srcVal.Index(i)
		str := elem.String()
		offset := i * size

		// Copy string bytes, pad or truncate as needed
		strBytes := []byte(str)
		copyLen := len(strBytes)
		if copyLen > size {
			copyLen = size
		}
		copy(data[offset:offset+copyLen], strBytes)

		// Handle padding based on string padding type
		switch dt.StringPadding {
		case message.PadNullTerm:
			// Ensure null termination if space allows
			if copyLen < size {
				data[offset+copyLen] = 0
			}
		case message.PadNullPad:
			// Remaining bytes are already zero (from make)
		case message.PadSpacePad:
			// Pad with spaces
			for j := copyLen; j < size; j++ {
				data[offset+j] = ' '
			}
		}
	}

	return data, nil
}

// encodeVarLen encodes variable-length strings.
// For variable-length strings, we return the actual string bytes directly.
// The hvl_t structure (offset + length) is handled at a higher level when writing.
func encodeVarLen(dt *message.Datatype, srcVal reflect.Value) ([]byte, error) {
	var n int
	switch srcVal.Kind() {
	case reflect.Slice, reflect.Array:
		n = srcVal.Len()
	default:
		n = 1
		sliceVal := reflect.MakeSlice(reflect.SliceOf(srcVal.Type()), 1, 1)
		sliceVal.Index(0).Set(srcVal)
		srcVal = sliceVal
	}

	var data []byte
	for i := 0; i < n; i++ {
		elem := srcVal.Index(i)
		if elem.Kind() == reflect.String {
			str := elem.String()
			data = append(data, []byte(str)...)
			data = append(data, 0)
		} else {
			return nil, fmt.Errorf("cannot encode %v as varlen string", elem.Kind())
		}
	}

	return data, nil
}

// GoTypeToDatatype creates an HDF5 datatype from a Go type.
func GoTypeToDatatype(t reflect.Type) (*message.Datatype, error) {
	// Handle pointer types
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
	}

	// Handle slice/array element types
	if t.Kind() == reflect.Slice || t.Kind() == reflect.Array {
		t = t.Elem()
	}

	switch t.Kind() {
	case reflect.Int8:
		return message.NewFixedPointDatatype(1, true, message.OrderLE), nil
	case reflect.Int16:
		return message.NewFixedPointDatatype(2, true, message.OrderLE), nil
	case reflect.Int32:
		return message.NewFixedPointDatatype(4, true, message.OrderLE), nil
	case reflect.Int64, reflect.Int:
		return message.NewFixedPointDatatype(8, true, message.OrderLE), nil
	case reflect.Uint8:
		return message.NewFixedPointDatatype(1, false, message.OrderLE), nil
	case reflect.Uint16:
		return message.NewFixedPointDatatype(2, false, message.OrderLE), nil
	case reflect.Uint32:
		return message.NewFixedPointDatatype(4, false, message.OrderLE), nil
	case reflect.Uint64, reflect.Uint:
		return message.NewFixedPointDatatype(8, false, message.OrderLE), nil
	case reflect.Float32:
		return message.NewFloatDatatype(4, message.OrderLE), nil
	case reflect.Float64:
		return message.NewFloatDatatype(8, message.OrderLE), nil
	case reflect.String:
		// Use variable-length string (default in h5py)
		return message.NewVarLenStringDatatype(message.CharsetUTF8), nil
	default:
		return nil, fmt.Errorf("unsupported Go type: %v", t)
	}
}

// DataSize returns the total size in bytes needed to store n elements of the given datatype.
func DataSize(dt *message.Datatype, n uint64) uint64 {
	return uint64(dt.Size) * n
}
