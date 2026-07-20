package hdf5

import (
	"fmt"
	"path"
	"reflect"

	"github.com/huangzhengshun/go-hdf5/internal/dtype"
	"github.com/huangzhengshun/go-hdf5/internal/layout"
	"github.com/huangzhengshun/go-hdf5/internal/message"
	"github.com/huangzhengshun/go-hdf5/internal/object"
)

// Dataset represents an HDF5 dataset.
type Dataset struct {
	file      *File
	path      string
	header    *object.Header
	dataspace *message.Dataspace
	datatype  *message.Datatype
	layout    layout.Layout

	// Write support fields
	dataAddr    uint64 // Address where data is stored
	dataSize    uint64 // Size of data in bytes
	numElements uint64 // Number of elements
}

// newDataset creates a Dataset from an object header.
func newDataset(f *File, path string, header *object.Header) (*Dataset, error) {
	ds := &Dataset{
		file:   f,
		path:   path,
		header: header,
	}

	// Get dataspace
	ds.dataspace = header.Dataspace()
	if ds.dataspace == nil {
		return nil, fmt.Errorf("dataset missing dataspace message")
	}

	// Get datatype
	ds.datatype = header.Datatype()
	if ds.datatype == nil {
		return nil, fmt.Errorf("dataset missing datatype message")
	}

	// Get layout
	layoutMsg := header.DataLayout()
	if layoutMsg == nil {
		return nil, fmt.Errorf("dataset missing layout message")
	}

	// Create layout handler
	filterMsg := header.FilterPipeline()
	var err error
	ds.layout, err = layout.New(layoutMsg, ds.dataspace, ds.datatype, filterMsg, f.reader)
	if err != nil {
		return nil, fmt.Errorf("creating layout: %w", err)
	}

	return ds, nil
}

// Name returns the dataset name (last component of path).
func (d *Dataset) Name() string {
	return path.Base(d.path)
}

// Path returns the full path to this dataset.
func (d *Dataset) Path() string {
	return d.path
}

// Shape returns the dimensions of the dataset.
func (d *Dataset) Shape() []uint64 {
	if d.dataspace.IsScalar() {
		return nil
	}
	return d.dataspace.Dimensions
}

// Rank returns the number of dimensions.
func (d *Dataset) Rank() int {
	return d.dataspace.Rank
}

// NumElements returns the total number of elements.
func (d *Dataset) NumElements() uint64 {
	return d.dataspace.NumElements()
}

// IsScalar returns true if the dataset is a scalar (single value).
func (d *Dataset) IsScalar() bool {
	return d.dataspace.IsScalar()
}

// DtypeSize returns the size of each element in bytes.
func (d *Dataset) DtypeSize() int {
	return int(d.datatype.Size)
}

// DtypeClass returns the datatype class.
func (d *Dataset) DtypeClass() message.DatatypeClass {
	return d.datatype.Class
}

// GoType returns the Go type that corresponds to this dataset's datatype.
func (d *Dataset) GoType() (reflect.Type, error) {
	return dtype.GoType(d.datatype)
}

// Read reads all data from the dataset into dest.
// dest should be a pointer to a slice of the appropriate type.
func (d *Dataset) Read(dest interface{}) error {
	// Read raw data
	raw, err := d.layout.Read()
	if err != nil {
		return fmt.Errorf("reading data: %w", err)
	}

	// Convert to Go types
	numElements := d.dataspace.NumElements()
	return dtype.Convert(d.datatype, raw, numElements, dest)
}

// ReadRaw reads all data from the dataset as raw bytes.
func (d *Dataset) ReadRaw() ([]byte, error) {
	return d.layout.Read()
}

// ReadSlice reads a hyperslab (rectangular selection) of the dataset.
// start specifies the starting coordinates, count specifies the number of elements per dimension.
// dest should be a pointer to a slice of the appropriate type.
//
// Example for a 2D dataset (10x20):
//
//	// Read a 5x10 region starting at (2, 5)
//	var result []float64
//	err := ds.ReadSlice([]uint64{2, 5}, []uint64{5, 10}, &result)
func (d *Dataset) ReadSlice(start, count []uint64, dest interface{}) error {
	// Read raw slice data
	raw, err := d.layout.ReadSlice(start, count)
	if err != nil {
		return fmt.Errorf("reading slice: %w", err)
	}

	// Calculate number of elements in the slice
	numElements := uint64(1)
	for _, c := range count {
		numElements *= c
	}

	// Convert to Go types
	return dtype.Convert(d.datatype, raw, numElements, dest)
}

// ReadSliceRaw reads a hyperslab as raw bytes without type conversion.
func (d *Dataset) ReadSliceRaw(start, count []uint64) ([]byte, error) {
	return d.layout.ReadSlice(start, count)
}

// ReadFloat64 reads the dataset as float64 values.
func (d *Dataset) ReadFloat64() ([]float64, error) {
	var result []float64
	err := d.Read(&result)
	return result, err
}

// ReadFloat32 reads the dataset as float32 values.
func (d *Dataset) ReadFloat32() ([]float32, error) {
	var result []float32
	err := d.Read(&result)
	return result, err
}

// ReadInt64 reads the dataset as int64 values.
func (d *Dataset) ReadInt64() ([]int64, error) {
	var result []int64
	err := d.Read(&result)
	return result, err
}

// ReadInt32 reads the dataset as int32 values.
func (d *Dataset) ReadInt32() ([]int32, error) {
	var result []int32
	err := d.Read(&result)
	return result, err
}

// ReadString reads the dataset as string values.
func (d *Dataset) ReadString() ([]string, error) {
	var result []string
	err := d.Read(&result)
	return result, err
}

// ReadInt8 reads the dataset as int8 values.
func (d *Dataset) ReadInt8() ([]int8, error) {
	var result []int8
	err := d.Read(&result)
	return result, err
}

// ReadInt16 reads the dataset as int16 values.
func (d *Dataset) ReadInt16() ([]int16, error) {
	var result []int16
	err := d.Read(&result)
	return result, err
}

// ReadUint8 reads the dataset as uint8 values.
func (d *Dataset) ReadUint8() ([]uint8, error) {
	var result []uint8
	err := d.Read(&result)
	return result, err
}

// ReadUint16 reads the dataset as uint16 values.
func (d *Dataset) ReadUint16() ([]uint16, error) {
	var result []uint16
	err := d.Read(&result)
	return result, err
}

// ReadUint32 reads the dataset as uint32 values.
func (d *Dataset) ReadUint32() ([]uint32, error) {
	var result []uint32
	err := d.Read(&result)
	return result, err
}

// ReadUint64 reads the dataset as uint64 values.
func (d *Dataset) ReadUint64() ([]uint64, error) {
	var result []uint64
	err := d.Read(&result)
	return result, err
}

// Attrs returns the attribute names for this dataset.
func (d *Dataset) Attrs() []string {
	var names []string
	for _, msg := range d.header.GetMessages(message.TypeAttribute) {
		attr := msg.(*message.Attribute)
		names = append(names, attr.Name)
	}
	return names
}

// Attr returns an attribute by name, or nil if not found.
func (d *Dataset) Attr(name string) *Attribute {
	for _, msg := range d.header.GetMessages(message.TypeAttribute) {
		attr := msg.(*message.Attribute)
		if attr.Name == name {
			return &Attribute{msg: attr, reader: d.file.reader}
		}
	}
	return nil
}

// HasAttr returns true if the dataset has an attribute with the given name.
func (d *Dataset) HasAttr(name string) bool {
	return d.Attr(name) != nil
}
