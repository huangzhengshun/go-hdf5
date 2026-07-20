package hdf5

import (
	"fmt"

	"github.com/huangzhengshun/go-hdf5/internal/binary"
	"github.com/huangzhengshun/go-hdf5/internal/dtype"
	"github.com/huangzhengshun/go-hdf5/internal/message"
)

// Attribute represents an HDF5 attribute attached to a dataset or group.
type Attribute struct {
	msg    *message.Attribute
	reader *binary.Reader // For resolving global heap references
}

// Name returns the attribute name.
func (a *Attribute) Name() string {
	return a.msg.Name
}

// Shape returns the dimensions of the attribute value.
func (a *Attribute) Shape() []uint64 {
	if a.msg.Dataspace == nil || a.msg.Dataspace.IsScalar() {
		return nil
	}
	return a.msg.Dataspace.Dimensions
}

// NumElements returns the total number of elements.
func (a *Attribute) NumElements() uint64 {
	if a.msg.Dataspace == nil {
		return 1
	}
	return a.msg.Dataspace.NumElements()
}

// IsScalar returns true if the attribute is a scalar value.
func (a *Attribute) IsScalar() bool {
	if a.msg.Dataspace == nil {
		return true
	}
	return a.msg.Dataspace.IsScalar()
}

// DtypeClass returns the datatype class.
func (a *Attribute) DtypeClass() message.DatatypeClass {
	if a.msg.Datatype == nil {
		return 0
	}
	return a.msg.Datatype.Class
}

// Read reads the attribute value into dest.
// dest should be a pointer to the appropriate type.
func (a *Attribute) Read(dest interface{}) error {
	if a.msg.Datatype == nil {
		return fmt.Errorf("attribute has no datatype")
	}
	if a.msg.Data == nil {
		return fmt.Errorf("attribute has no data")
	}

	numElements := a.NumElements()
	return dtype.ConvertWithReader(a.msg.Datatype, a.msg.Data, numElements, dest, a.reader)
}

// ReadFloat64 reads the attribute as float64 values.
func (a *Attribute) ReadFloat64() ([]float64, error) {
	var result []float64
	err := a.Read(&result)
	return result, err
}

// ReadFloat32 reads the attribute as float32 values.
func (a *Attribute) ReadFloat32() ([]float32, error) {
	var result []float32
	err := a.Read(&result)
	return result, err
}

// ReadInt64 reads the attribute as int64 values.
func (a *Attribute) ReadInt64() ([]int64, error) {
	var result []int64
	err := a.Read(&result)
	return result, err
}

// ReadInt32 reads the attribute as int32 values.
func (a *Attribute) ReadInt32() ([]int32, error) {
	var result []int32
	err := a.Read(&result)
	return result, err
}

// ReadString reads the attribute as string values.
func (a *Attribute) ReadString() ([]string, error) {
	var result []string
	err := a.Read(&result)
	return result, err
}

// ReadScalarInt64 reads a scalar int64 attribute.
func (a *Attribute) ReadScalarInt64() (int64, error) {
	vals, err := a.ReadInt64()
	if err != nil {
		return 0, err
	}
	if len(vals) == 0 {
		return 0, fmt.Errorf("no values in attribute")
	}
	return vals[0], nil
}

// ReadScalarFloat64 reads a scalar float64 attribute.
func (a *Attribute) ReadScalarFloat64() (float64, error) {
	vals, err := a.ReadFloat64()
	if err != nil {
		return 0, err
	}
	if len(vals) == 0 {
		return 0, fmt.Errorf("no values in attribute")
	}
	return vals[0], nil
}

// ReadScalarString reads a scalar string attribute.
func (a *Attribute) ReadScalarString() (string, error) {
	vals, err := a.ReadString()
	if err != nil {
		return "", err
	}
	if len(vals) == 0 {
		return "", fmt.Errorf("no values in attribute")
	}
	return vals[0], nil
}

// ReadCompound reads the attribute as compound type values.
// Returns a slice of map[string]interface{} with member names as keys.
func (a *Attribute) ReadCompound() ([]map[string]interface{}, error) {
	var result []interface{}
	err := a.Read(&result)
	if err != nil {
		return nil, err
	}

	// Convert []interface{} to []map[string]interface{}
	maps := make([]map[string]interface{}, len(result))
	for i, v := range result {
		if m, ok := v.(map[string]interface{}); ok {
			maps[i] = m
		} else {
			return nil, fmt.Errorf("element %d is not a map: %T", i, v)
		}
	}
	return maps, nil
}

// ReadScalarCompound reads a scalar compound attribute.
// Returns a map[string]interface{} with member names as keys.
func (a *Attribute) ReadScalarCompound() (map[string]interface{}, error) {
	vals, err := a.ReadCompound()
	if err != nil {
		return nil, err
	}
	if len(vals) == 0 {
		return nil, fmt.Errorf("no values in attribute")
	}
	return vals[0], nil
}

// ReadArray reads the attribute value which is an array type.
// Returns the array data as interface{} (the actual type depends on the base type).
func (a *Attribute) ReadArray() (interface{}, error) {
	var result interface{}
	err := a.Read(&result)
	return result, err
}

// IsCompound returns true if the attribute has a compound datatype.
func (a *Attribute) IsCompound() bool {
	return a.msg.Datatype != nil && a.msg.Datatype.Class == message.ClassCompound
}

// IsArray returns true if the attribute has an array datatype.
func (a *Attribute) IsArray() bool {
	return a.msg.Datatype != nil && a.msg.Datatype.Class == message.ClassArray
}

// Value reads the attribute and returns an auto-typed Go value.
// Returns appropriate types based on HDF5 datatype:
//   - Fixed-point (integers): int64 or []int64
//   - Floating-point: float64 or []float64
//   - String: string or []string
//   - Compound: map[string]interface{} or []map[string]interface{}
//   - Array: the base type as a slice
//
// For scalar attributes, returns a single value. For array dataspaces,
// returns a slice.
func (a *Attribute) Value() (interface{}, error) {
	if a.msg.Datatype == nil {
		return nil, fmt.Errorf("attribute has no datatype")
	}

	isScalar := a.IsScalar()
	class := a.msg.Datatype.Class

	switch class {
	case message.ClassFixedPoint:
		if a.msg.Datatype.Signed {
			vals, err := a.ReadInt64()
			if err != nil {
				return nil, err
			}
			if isScalar && len(vals) == 1 {
				return vals[0], nil
			}
			return vals, nil
		}
		// Unsigned - read as int64 and convert to uint64
		var vals []uint64
		if err := a.Read(&vals); err != nil {
			return nil, err
		}
		if isScalar && len(vals) == 1 {
			return vals[0], nil
		}
		return vals, nil

	case message.ClassFloatPoint:
		vals, err := a.ReadFloat64()
		if err != nil {
			return nil, err
		}
		if isScalar && len(vals) == 1 {
			return vals[0], nil
		}
		return vals, nil

	case message.ClassString:
		vals, err := a.ReadString()
		if err != nil {
			return nil, err
		}
		if isScalar && len(vals) == 1 {
			return vals[0], nil
		}
		return vals, nil

	case message.ClassVarLen:
		if a.msg.Datatype.IsVarLenString {
			vals, err := a.ReadString()
			if err != nil {
				return nil, err
			}
			if isScalar && len(vals) == 1 {
				return vals[0], nil
			}
			return vals, nil
		}
		// Variable-length sequence - read as generic
		var result interface{}
		if err := a.Read(&result); err != nil {
			return nil, err
		}
		return result, nil

	case message.ClassCompound:
		vals, err := a.ReadCompound()
		if err != nil {
			return nil, err
		}
		if isScalar && len(vals) == 1 {
			return vals[0], nil
		}
		return vals, nil

	case message.ClassArray:
		var result interface{}
		if err := a.Read(&result); err != nil {
			return nil, err
		}
		return result, nil

	case message.ClassEnum:
		// Enums are typically stored as integers
		vals, err := a.ReadInt64()
		if err != nil {
			return nil, err
		}
		if isScalar && len(vals) == 1 {
			return vals[0], nil
		}
		return vals, nil

	default:
		// Fallback: try generic read
		var result interface{}
		if err := a.Read(&result); err != nil {
			return nil, err
		}
		return result, nil
	}
}
