// Package dtype provides HDF5 datatype handling and Go type conversion.
//
// This package bridges the gap between HDF5's type system and Go's type system,
// providing functionality to:
//
//   - Determine the Go type corresponding to an HDF5 datatype
//   - Convert raw HDF5 data bytes to Go values
//   - Encode Go values to HDF5 data bytes
//   - Create HDF5 datatypes from Go types
//
// # Type Mapping Strategy
//
// HDF5 datatypes are mapped to Go types as follows:
//
//	HDF5 Class        | Go Type
//	------------------|------------------
//	Fixed-point (int) | int8/16/32/64 or uint8/16/32/64 based on size and signedness
//	Floating-point    | float32 (4 bytes) or float64 (8 bytes)
//	String (fixed)    | string
//	String (varlen)   | string (via global heap lookup)
//	Compound          | map[string]interface{} or struct
//	Array             | slice of element type
//	Enum              | underlying integer type
//	Bitfield          | unsigned integer type
//	Opaque            | []byte
//
// # Reading Data
//
// Use [Convert] or [ConvertWithReader] to convert raw bytes to Go values:
//
//	var values []float64
//	err := dtype.Convert(datatype, rawBytes, numElements, &values)
//
// For variable-length data (like varlen strings), pass a reader to access
// the global heap:
//
//	err := dtype.ConvertWithReader(datatype, rawBytes, n, &values, reader)
//
// # Writing Data
//
// Use [Encode] to convert Go values to raw bytes:
//
//	data, err := dtype.Encode(datatype, []int32{1, 2, 3})
//
// Use [GoTypeToDatatype] to create an HDF5 datatype from a Go type:
//
//	dt, err := dtype.GoTypeToDatatype(reflect.TypeOf([]float64{}))
//
// # Key Functions
//
//   - [GoType]: Returns the reflect.Type for an HDF5 datatype
//   - [Convert]: Converts HDF5 bytes to Go values
//   - [ConvertWithReader]: Converts with reader access for varlen data
//   - [Encode]: Converts Go values to HDF5 bytes
//   - [GoTypeToDatatype]: Creates HDF5 datatype from Go type
//   - [ByteOrder]: Returns the binary.ByteOrder for a datatype
//   - [ElementSize]: Returns the size of a single element in bytes
package dtype
