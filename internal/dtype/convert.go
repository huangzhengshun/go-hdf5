package dtype

// Type Conversion Strategy
//
// This file implements conversion between HDF5 raw bytes and Go values.
// The strategy handles both reading (HDF5 -> Go) and the data structures
// needed for efficient conversion.
//
// # Conversion Dispatch
//
// The main Convert function dispatches based on the HDF5 datatype class:
//
//   - Fixed-point (integers): Converts using byte order and size
//   - Float-point: Converts using IEEE 754 bit representations
//   - String (fixed): Copies bytes, handles null/space padding
//   - String (varlen): Resolves global heap references
//   - Compound: Recursively converts each member by offset
//   - Array: Converts element sequences based on array dimensions
//   - Enum: Converts as underlying integer type
//   - Bitfield: Converts as unsigned integer
//   - Opaque: Returns raw bytes
//
// # Fast Path Optimization
//
// For common cases where the HDF5 type exactly matches the Go type (same
// size, same endianness as the platform), we use direct memory copy via
// unsafe.Pointer. This is controlled by canDirectCopy() and directCopy().
//
// The fast path applies when:
//   - Byte order is little-endian (matches x86/ARM platforms)
//   - Element size matches the Go type size
//   - Type class is fixed-point or float-point
//
// # Variable-Length Data
//
// Variable-length strings and sequences store references to the global heap
// rather than inline data. Each reference contains:
//   - 4 bytes: sequence length
//   - offsetSize bytes: global heap collection address
//   - 4 bytes: object index within collection
//
// The convertVarLenString function resolves these references by reading
// from the global heap, caching collections for efficiency.
//
// # Compound Type Handling
//
// Compound types (structs) store members at specific byte offsets within
// each element. The conversion extracts each member's bytes based on its
// offset and recursively converts using the member's datatype.

import (
	"fmt"
	"math"
	"reflect"
	"unsafe"

	"github.com/huangzhengshun/go-hdf5/internal/binary"
	"github.com/huangzhengshun/go-hdf5/internal/heap"
	"github.com/huangzhengshun/go-hdf5/internal/message"
)

// Convert converts raw HDF5 data to Go values.
// The dest parameter should be a pointer to a slice or array of the appropriate type.
func Convert(dt *message.Datatype, data []byte, numElements uint64, dest interface{}) error {
	return ConvertWithReader(dt, data, numElements, dest, nil)
}

// ConvertWithReader converts raw HDF5 data to Go values, with access to a reader
// for resolving global heap references (needed for variable-length data).
func ConvertWithReader(dt *message.Datatype, data []byte, numElements uint64, dest interface{}, reader *binary.Reader) error {
	if dt == nil {
		return fmt.Errorf("nil datatype")
	}

	destVal := reflect.ValueOf(dest)
	if destVal.Kind() != reflect.Ptr {
		return fmt.Errorf("dest must be a pointer")
	}

	elemVal := destVal.Elem()

	switch dt.Class {
	case message.ClassFixedPoint:
		return convertFixedPoint(dt, data, numElements, elemVal)
	case message.ClassFloatPoint:
		return convertFloatPoint(dt, data, numElements, elemVal)
	case message.ClassString:
		return convertString(dt, data, numElements, elemVal)
	case message.ClassVarLen:
		return convertVarLen(dt, data, numElements, elemVal, reader)
	case message.ClassCompound:
		return convertCompound(dt, data, numElements, elemVal, reader)
	case message.ClassArray:
		return convertArray(dt, data, numElements, elemVal, reader)
	case message.ClassEnum:
		return convertEnum(dt, data, numElements, elemVal)
	case message.ClassBitfield:
		return convertBitfield(dt, data, numElements, elemVal)
	case message.ClassOpaque:
		return convertOpaque(dt, data, numElements, elemVal)
	default:
		return fmt.Errorf("unsupported datatype class for conversion: %d", dt.Class)
	}
}

// ConvertToSlice converts raw HDF5 data to a newly allocated slice.
func ConvertToSlice[T any](dt *message.Datatype, data []byte, numElements uint64) ([]T, error) {
	result := make([]T, numElements)
	err := Convert(dt, data, numElements, &result)
	if err != nil {
		return nil, err
	}
	return result, nil
}

func convertFixedPoint(dt *message.Datatype, data []byte, n uint64, dest reflect.Value) error {
	order := ByteOrder(dt)
	size := int(dt.Size)
	signed := dt.Signed

	// Fast path: if dest is a compatible slice and endianness matches
	if dest.Kind() == reflect.Slice && dest.CanSet() {
		if canDirectCopy(dt, dest.Type().Elem()) {
			return directCopy(data, n, size, dest)
		}
	}

	// Slow path: element-by-element conversion
	if dest.Kind() == reflect.Slice {
		if dest.Len() < int(n) {
			dest.Set(reflect.MakeSlice(dest.Type(), int(n), int(n)))
		}
	}

	for i := uint64(0); i < n; i++ {
		offset := int(i) * size
		if offset+size > len(data) {
			break
		}

		elemData := data[offset : offset+size]
		var val interface{}

		switch size {
		case 1:
			if signed {
				val = int8(elemData[0])
			} else {
				val = elemData[0]
			}
		case 2:
			v := order.Uint16(elemData)
			if signed {
				val = int16(v)
			} else {
				val = v
			}
		case 4:
			v := order.Uint32(elemData)
			if signed {
				val = int32(v)
			} else {
				val = v
			}
		case 8:
			v := order.Uint64(elemData)
			if signed {
				val = int64(v)
			} else {
				val = v
			}
		default:
			return fmt.Errorf("unsupported integer size: %d", size)
		}

		if dest.Kind() == reflect.Slice {
			elemType := dest.Type().Elem()
			if elemType.Kind() == reflect.Bool {
				if uintVal, ok := val.(uint8); ok {
					dest.Index(int(i)).SetBool(uintVal != 0)
				} else {
					dest.Index(int(i)).SetBool(val != nil)
				}
			} else {
				dest.Index(int(i)).Set(reflect.ValueOf(val).Convert(elemType))
			}
		} else if dest.Kind() == reflect.Bool {
			if uintVal, ok := val.(uint8); ok {
				dest.SetBool(uintVal != 0)
			} else {
				dest.SetBool(val != nil)
			}
		} else if i == 0 {
			dest.Set(reflect.ValueOf(val).Convert(dest.Type()))
		}
	}

	return nil
}

func convertFloatPoint(dt *message.Datatype, data []byte, n uint64, dest reflect.Value) error {
	order := ByteOrder(dt)
	size := int(dt.Size)

	// Fast path
	if dest.Kind() == reflect.Slice && canDirectCopy(dt, dest.Type().Elem()) {
		return directCopy(data, n, size, dest)
	}

	// Slow path
	if dest.Kind() == reflect.Slice {
		if dest.Len() < int(n) {
			dest.Set(reflect.MakeSlice(dest.Type(), int(n), int(n)))
		}
	}

	for i := uint64(0); i < n; i++ {
		offset := int(i) * size
		if offset+size > len(data) {
			break
		}

		elemData := data[offset : offset+size]
		var val interface{}

		switch size {
		case 4:
			bits := order.Uint32(elemData)
			val = math.Float32frombits(bits)
		case 8:
			bits := order.Uint64(elemData)
			val = math.Float64frombits(bits)
		default:
			return fmt.Errorf("unsupported float size: %d", size)
		}

		if dest.Kind() == reflect.Slice {
			dest.Index(int(i)).Set(reflect.ValueOf(val).Convert(dest.Type().Elem()))
		} else if i == 0 {
			dest.Set(reflect.ValueOf(val).Convert(dest.Type()))
		}
	}

	return nil
}

func convertString(dt *message.Datatype, data []byte, n uint64, dest reflect.Value) error {
	size := int(dt.Size)

	if dest.Kind() == reflect.Slice {
		if dest.Len() < int(n) {
			dest.Set(reflect.MakeSlice(dest.Type(), int(n), int(n)))
		}
	}

	for i := uint64(0); i < n; i++ {
		offset := int(i) * size
		if offset+size > len(data) {
			break
		}

		strData := data[offset : offset+size]

		// Find null terminator or end of padding
		end := len(strData)
		for j := 0; j < len(strData); j++ {
			if strData[j] == 0 {
				end = j
				break
			}
		}

		// Trim trailing spaces for space-padded strings
		if dt.StringPadding == message.PadSpacePad {
			for end > 0 && strData[end-1] == ' ' {
				end--
			}
		}

		str := string(strData[:end])

		if dest.Kind() == reflect.Slice {
			dest.Index(int(i)).SetString(str)
		} else if dest.Kind() == reflect.String {
			dest.SetString(str)
		}
	}

	return nil
}

func convertVarLen(dt *message.Datatype, data []byte, n uint64, dest reflect.Value, reader *binary.Reader) error {
	// Variable-length data references the global heap
	if dt.IsVarLenString {
		return convertVarLenString(dt, data, n, dest, reader)
	}

	return fmt.Errorf("variable-length data type not fully supported (IsVarLenString=%v)", dt.IsVarLenString)
}

func convertVarLenString(dt *message.Datatype, data []byte, n uint64, dest reflect.Value, reader *binary.Reader) error {
	// Variable-length strings are stored as:
	// - 4 bytes: sequence length (number of characters)
	// - offsetSize bytes: heap collection address
	// - 4 bytes: object index
	//
	// Each reference is (4 + offsetSize + 4) bytes total.

	if dest.Kind() == reflect.Slice {
		if dest.Len() < int(n) {
			dest.Set(reflect.MakeSlice(dest.Type(), int(n), int(n)))
		}
	}

	// Determine offset size - if no reader, try to infer from data size
	offsetSize := 8 // Default to 8
	if reader != nil {
		offsetSize = reader.OffsetSize()
	}

	refSize := 4 + offsetSize + 4 // length + address + index
	heapCache := make(map[uint64]*heap.GlobalHeap)

	for i := uint64(0); i < n; i++ {
		offset := int(i) * refSize
		if offset+refSize > len(data) {
			break
		}

		refData := data[offset : offset+refSize]

		// Skip the 4-byte sequence length, then parse the global heap ID
		// The sequence length at bytes 0-3 tells us the string length, but
		// the actual string data is in the global heap
		heapID, err := heap.ParseGlobalHeapID(refData[4:], offsetSize)
		if err != nil {
			return fmt.Errorf("parsing global heap ID for element %d: %w", i, err)
		}

		// Skip null references (address 0)
		if heapID.CollectionAddress == 0 {
			if dest.Kind() == reflect.Slice && dest.Type().Elem().Kind() == reflect.String {
				dest.Index(int(i)).SetString("")
			}
			continue
		}

		// We need the reader to access the global heap
		if reader == nil {
			return fmt.Errorf("variable-length string reading requires file reader (global heap at 0x%x)", heapID.CollectionAddress)
		}

		// Get or read the global heap collection (cache for efficiency)
		gh, ok := heapCache[heapID.CollectionAddress]
		if !ok {
			gh, err = heap.ReadGlobalHeap(reader, heapID.CollectionAddress)
			if err != nil {
				return fmt.Errorf("reading global heap at 0x%x: %w", heapID.CollectionAddress, err)
			}
			heapCache[heapID.CollectionAddress] = gh
		}

		// Get the string from the heap
		str, err := gh.GetString(uint16(heapID.ObjectIndex))
		if err != nil {
			return fmt.Errorf("getting string from heap (index %d): %w", heapID.ObjectIndex, err)
		}

		if dest.Kind() == reflect.Slice && dest.Type().Elem().Kind() == reflect.String {
			dest.Index(int(i)).SetString(str)
		} else if dest.Kind() == reflect.String && i == 0 {
			dest.SetString(str)
		}
	}

	return nil
}

func convertCompound(dt *message.Datatype, data []byte, n uint64, dest reflect.Value, reader *binary.Reader) error {
	// Compound types are stored as contiguous bytes with members at specific offsets
	size := int(dt.Size)

	if dest.Kind() == reflect.Slice {
		if dest.Len() < int(n) {
			dest.Set(reflect.MakeSlice(dest.Type(), int(n), int(n)))
		}
	}

	for i := uint64(0); i < n; i++ {
		offset := int(i) * size
		if offset+size > len(data) {
			break
		}

		elemData := data[offset : offset+size]
		result := make(map[string]interface{})

		for _, member := range dt.Members {
			memberOffset := int(member.ByteOffset)
			if member.Type == nil {
				continue
			}

			memberSize := int(member.Type.Size)
			if memberOffset+memberSize > len(elemData) {
				continue
			}

			memberData := elemData[memberOffset : memberOffset+memberSize]
			memberValue, err := convertMemberValue(member.Type, memberData, reader)
			if err != nil {
				return fmt.Errorf("converting compound member %q: %w", member.Name, err)
			}
			result[member.Name] = memberValue
		}

		if dest.Kind() == reflect.Slice {
			elemType := dest.Type().Elem()
			if elemType.Kind() == reflect.Map {
				dest.Index(int(i)).Set(reflect.ValueOf(result))
			} else if elemType.Kind() == reflect.Interface {
				dest.Index(int(i)).Set(reflect.ValueOf(result))
			}
		} else if dest.Kind() == reflect.Map && i == 0 {
			for k, v := range result {
				dest.SetMapIndex(reflect.ValueOf(k), reflect.ValueOf(v))
			}
		} else if dest.Kind() == reflect.Interface && i == 0 {
			dest.Set(reflect.ValueOf(result))
		}
	}

	return nil
}

// convertMemberValue converts a single compound member value.
func convertMemberValue(dt *message.Datatype, data []byte, reader *binary.Reader) (interface{}, error) {
	switch dt.Class {
	case message.ClassFixedPoint:
		order := ByteOrder(dt)
		size := int(dt.Size)
		switch size {
		case 1:
			if dt.Signed {
				return int8(data[0]), nil
			}
			return data[0], nil
		case 2:
			v := order.Uint16(data)
			if dt.Signed {
				return int16(v), nil
			}
			return v, nil
		case 4:
			v := order.Uint32(data)
			if dt.Signed {
				return int32(v), nil
			}
			return v, nil
		case 8:
			v := order.Uint64(data)
			if dt.Signed {
				return int64(v), nil
			}
			return v, nil
		}
	case message.ClassFloatPoint:
		order := ByteOrder(dt)
		size := int(dt.Size)
		switch size {
		case 4:
			return math.Float32frombits(order.Uint32(data)), nil
		case 8:
			return math.Float64frombits(order.Uint64(data)), nil
		}
	case message.ClassString:
		size := int(dt.Size)
		if size > len(data) {
			size = len(data)
		}
		end := size
		for j := 0; j < size; j++ {
			if data[j] == 0 {
				end = j
				break
			}
		}
		return string(data[:end]), nil
	case message.ClassCompound:
		result := make(map[string]interface{})
		for _, member := range dt.Members {
			memberOffset := int(member.ByteOffset)
			if member.Type == nil {
				continue
			}
			memberSize := int(member.Type.Size)
			if memberOffset+memberSize > len(data) {
				continue
			}
			memberData := data[memberOffset : memberOffset+memberSize]
			val, err := convertMemberValue(member.Type, memberData, reader)
			if err != nil {
				return nil, err
			}
			result[member.Name] = val
		}
		return result, nil
	}
	return nil, fmt.Errorf("unsupported member type class: %d", dt.Class)
}

func convertArray(dt *message.Datatype, data []byte, n uint64, dest reflect.Value, reader *binary.Reader) error {
	// Array types store fixed-size arrays of the base type
	if dt.BaseType == nil || len(dt.ArrayDims) == 0 {
		return fmt.Errorf("invalid array type: missing base type or dimensions")
	}

	// Calculate total elements in the array
	arrayElements := uint64(1)
	for _, dim := range dt.ArrayDims {
		arrayElements *= uint64(dim)
	}

	size := int(dt.Size)
	baseSize := int(dt.BaseType.Size)

	if dest.Kind() == reflect.Slice {
		if dest.Len() < int(n) {
			dest.Set(reflect.MakeSlice(dest.Type(), int(n), int(n)))
		}
	}

	for i := uint64(0); i < n; i++ {
		offset := int(i) * size
		if offset+size > len(data) {
			break
		}

		elemData := data[offset : offset+size]

		// Convert the array elements based on base type
		var arrayResult interface{}
		switch dt.BaseType.Class {
		case message.ClassFixedPoint:
			if dt.BaseType.Signed {
				switch baseSize {
				case 4:
					arr := make([]int32, arrayElements)
					for j := uint64(0); j < arrayElements; j++ {
						order := ByteOrder(dt.BaseType)
						arr[j] = int32(order.Uint32(elemData[j*4:]))
					}
					arrayResult = arr
				case 8:
					arr := make([]int64, arrayElements)
					for j := uint64(0); j < arrayElements; j++ {
						order := ByteOrder(dt.BaseType)
						arr[j] = int64(order.Uint64(elemData[j*8:]))
					}
					arrayResult = arr
				default:
					return fmt.Errorf("unsupported array element size: %d", baseSize)
				}
			} else {
				switch baseSize {
				case 4:
					arr := make([]uint32, arrayElements)
					for j := uint64(0); j < arrayElements; j++ {
						order := ByteOrder(dt.BaseType)
						arr[j] = order.Uint32(elemData[j*4:])
					}
					arrayResult = arr
				case 8:
					arr := make([]uint64, arrayElements)
					for j := uint64(0); j < arrayElements; j++ {
						order := ByteOrder(dt.BaseType)
						arr[j] = order.Uint64(elemData[j*8:])
					}
					arrayResult = arr
				default:
					return fmt.Errorf("unsupported array element size: %d", baseSize)
				}
			}
		case message.ClassFloatPoint:
			switch baseSize {
			case 4:
				arr := make([]float32, arrayElements)
				for j := uint64(0); j < arrayElements; j++ {
					order := ByteOrder(dt.BaseType)
					arr[j] = math.Float32frombits(order.Uint32(elemData[j*4:]))
				}
				arrayResult = arr
			case 8:
				arr := make([]float64, arrayElements)
				for j := uint64(0); j < arrayElements; j++ {
					order := ByteOrder(dt.BaseType)
					arr[j] = math.Float64frombits(order.Uint64(elemData[j*8:]))
				}
				arrayResult = arr
			default:
				return fmt.Errorf("unsupported array float size: %d", baseSize)
			}
		default:
			return fmt.Errorf("unsupported array base type: %d", dt.BaseType.Class)
		}

		if dest.Kind() == reflect.Slice {
			dest.Index(int(i)).Set(reflect.ValueOf(arrayResult))
		} else if dest.Kind() == reflect.Interface && i == 0 {
			dest.Set(reflect.ValueOf(arrayResult))
		}
	}

	return nil
}

func convertEnum(dt *message.Datatype, data []byte, n uint64, dest reflect.Value) error {
	// Enums are stored as their underlying integer type
	// For now, convert to the base integer type
	order := ByteOrder(dt)
	size := int(dt.Size)

	if dest.Kind() == reflect.Slice {
		if dest.Len() < int(n) {
			dest.Set(reflect.MakeSlice(dest.Type(), int(n), int(n)))
		}
	}

	for i := uint64(0); i < n; i++ {
		offset := int(i) * size
		if offset+size > len(data) {
			break
		}

		elemData := data[offset : offset+size]
		var val interface{}

		switch size {
		case 1:
			val = int32(int8(elemData[0]))
		case 2:
			val = int32(int16(order.Uint16(elemData)))
		case 4:
			val = int32(order.Uint32(elemData))
		case 8:
			val = int64(order.Uint64(elemData))
		default:
			return fmt.Errorf("unsupported enum size: %d", size)
		}

		if dest.Kind() == reflect.Slice {
			dest.Index(int(i)).Set(reflect.ValueOf(val).Convert(dest.Type().Elem()))
		}
	}

	return nil
}

func convertBitfield(dt *message.Datatype, data []byte, n uint64, dest reflect.Value) error {
	// Bitfields are stored as unsigned integers
	order := ByteOrder(dt)
	size := int(dt.Size)

	if dest.Kind() == reflect.Slice {
		if dest.Len() < int(n) {
			dest.Set(reflect.MakeSlice(dest.Type(), int(n), int(n)))
		}
	}

	for i := uint64(0); i < n; i++ {
		offset := int(i) * size
		if offset+size > len(data) {
			break
		}

		elemData := data[offset : offset+size]
		var val interface{}

		switch size {
		case 1:
			val = elemData[0]
		case 2:
			val = order.Uint16(elemData)
		case 4:
			val = order.Uint32(elemData)
		case 8:
			val = order.Uint64(elemData)
		default:
			return fmt.Errorf("unsupported bitfield size: %d", size)
		}

		if dest.Kind() == reflect.Slice {
			dest.Index(int(i)).Set(reflect.ValueOf(val).Convert(dest.Type().Elem()))
		}
	}

	return nil
}

func convertOpaque(dt *message.Datatype, data []byte, n uint64, dest reflect.Value) error {
	// Opaque types are returned as raw byte slices
	size := int(dt.Size)

	if dest.Kind() == reflect.Slice {
		if dest.Len() < int(n) {
			dest.Set(reflect.MakeSlice(dest.Type(), int(n), int(n)))
		}
	}

	for i := uint64(0); i < n; i++ {
		offset := int(i) * size
		if offset+size > len(data) {
			break
		}

		elemData := make([]byte, size)
		copy(elemData, data[offset:offset+size])

		if dest.Kind() == reflect.Slice {
			dest.Index(int(i)).Set(reflect.ValueOf(elemData))
		}
	}

	return nil
}

// canDirectCopy checks if we can do a direct memory copy.
func canDirectCopy(dt *message.Datatype, elemType reflect.Type) bool {
	// Must be little-endian (native for most systems)
	if dt.ByteOrder != message.OrderLE {
		return false
	}

	// Size must match
	if int(dt.Size) != int(elemType.Size()) {
		return false
	}

	// Type must be compatible
	switch dt.Class {
	case message.ClassFixedPoint:
		switch elemType.Kind() {
		case reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
			return dt.Signed
		case reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
			return !dt.Signed
		}
	case message.ClassFloatPoint:
		switch elemType.Kind() {
		case reflect.Float32, reflect.Float64:
			return true
		}
	}

	return false
}

// directCopy performs a direct memory copy for compatible types.
func directCopy(data []byte, n uint64, size int, dest reflect.Value) error {
	needed := int(n) * size
	if needed > len(data) {
		return fmt.Errorf("not enough data: need %d bytes, have %d", needed, len(data))
	}

	if dest.Len() < int(n) {
		dest.Set(reflect.MakeSlice(dest.Type(), int(n), int(n)))
	}

	// Get pointer to slice data
	sliceHeader := (*reflect.SliceHeader)(unsafe.Pointer(dest.UnsafeAddr()))
	destPtr := unsafe.Pointer(sliceHeader.Data)

	// Copy data directly
	copy(unsafe.Slice((*byte)(destPtr), needed), data[:needed])

	return nil
}

// ReadScalar reads a single scalar value from raw data.
func ReadScalar[T any](dt *message.Datatype, data []byte) (T, error) {
	var zero T
	result := make([]T, 1)
	err := Convert(dt, data, 1, &result)
	if err != nil {
		return zero, err
	}
	return result[0], nil
}
