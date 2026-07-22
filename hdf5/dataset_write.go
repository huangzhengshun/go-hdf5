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

// CreateDataset creates a new dataset with the given name, dimensions, and data type.
// The datatype is inferred from the provided Go type.
func (g *Group) CreateDataset(name string, data interface{}, opts ...DatasetOption) (*Dataset, error) {
	if !g.file.writable {
		return nil, fmt.Errorf("file is not writable")
	}

	if name == "" {
		return nil, fmt.Errorf("dataset name cannot be empty")
	}

	options := defaultDatasetOptions()
	for _, opt := range opts {
		if opt != nil {
			opt(options)
		}
	}

	// Get the data value and type
	dataVal := reflect.ValueOf(data)
	if dataVal.Kind() == reflect.Ptr {
		dataVal = dataVal.Elem()
	}

	// Determine dimensions and element type
	dims, elemType, err := inferDimensionsAndType(dataVal)
	if err != nil {
		return nil, fmt.Errorf("inferring dimensions: %w", err)
	}

	// Create datatype from Go type
	datatype, err := dtype.GoTypeToDatatype(elemType)
	if err != nil {
		return nil, fmt.Errorf("creating datatype: %w", err)
	}

	// Create dataspace
	dataspace := message.NewDataspace(dims, options.maxDims)

	// Calculate total number of elements
	numElements := uint64(1)
	for _, d := range dims {
		numElements *= d
	}

	// Encode the data
	rawData, err := dtype.Encode(datatype, data)
	if err != nil {
		return nil, fmt.Errorf("encoding data: %w", err)
	}

	// Determine layout
	var dataLayout *message.DataLayout

	if options.chunks != nil {
		// Chunked layout
		chunkDims := make([]uint32, len(options.chunks))
		for i, c := range options.chunks {
			chunkDims[i] = uint32(c)
		}

		// Create chunk writer
		cw := layout.NewChunkWriter(g.file.writer, chunkDims, datatype.Size, g.file.allocate)

		// Check if data fits in a single chunk
		chunkSize := cw.ChunkSize()
		dataSize := uint64(len(rawData))

		if dataSize <= chunkSize {
			// Single chunk - use Implicit index type (compatible with h5py)
			chunkAddr, err := cw.WriteSingleChunk(rawData)
			if err != nil {
				return nil, fmt.Errorf("writing chunk: %w", err)
			}

			// For Implicit index, the address points to the chunk data
			dataLayout = message.NewChunkedLayout(chunkDims, datatype.Size, message.ChunkIndexImplicit)
			dataLayout.ChunkIndexAddr = chunkAddr
		} else {
			// Multiple chunks - use Fixed Array (FAHD/FADB)
			// Note: h5py compatibility is limited for multi-chunk datasets
			chunks := layout.SplitIntoChunks(rawData, dims, chunkDims, datatype.Size)
			chunkAddrs, err := cw.WriteChunks(chunks)
			if err != nil {
				return nil, fmt.Errorf("writing chunks: %w", err)
			}

			// Write the fixed array index
			indexAddr, err := cw.WriteFixedArrayIndex(chunkAddrs, nil)
			if err != nil {
				return nil, fmt.Errorf("writing chunk index: %w", err)
			}

			dataLayout = message.NewChunkedLayout(chunkDims, datatype.Size, message.ChunkIndexFixedArray)
			dataLayout.ChunkIndexAddr = indexAddr
		}
	} else {
		// Contiguous layout
		dataSize := uint64(len(rawData))
		dataAddr := g.file.allocate(int64(dataSize))

		// Write the raw data
		w := g.file.writer.At(int64(dataAddr))
		if err := w.WriteBytes(rawData); err != nil {
			return nil, fmt.Errorf("writing data: %w", err)
		}

		dataLayout = message.NewContiguousLayout(dataAddr, dataSize)
	}

	// Create dataset object header
	messages := object.NewDatasetHeader(dataspace, datatype, dataLayout)

	// Add attributes if specified
	for _, attr := range options.attributes {
		attrMsg, err := createAttributeMessage(attr.name, attr.value)
		if err != nil {
			return nil, fmt.Errorf("creating attribute %q: %w", attr.name, err)
		}
		messages = append(messages, attrMsg)
	}

	// Calculate header size and allocate
	headerSize := object.HeaderSize(g.file.writer, messages)
	datasetAddr := g.file.allocate(int64(headerSize))

	// Write the dataset object header
	hw := g.file.writer.At(int64(datasetAddr))
	if _, err := object.WriteHeader(hw, messages); err != nil {
		return nil, fmt.Errorf("writing dataset header: %w", err)
	}

	// Create a hard link from parent group to this dataset
	link := message.NewHardLink(name, datasetAddr)
	if err := g.addLink(link); err != nil {
		return nil, fmt.Errorf("adding link to parent: %w", err)
	}

	// Calculate the path for the new dataset
	newPath := path.Join(g.path, name)
	if g.path == "/" {
		newPath = "/" + name
	}

	// Create the Dataset object
	ds := &Dataset{
		file:      g.file,
		path:      newPath,
		header:    nil, // Will be loaded on demand
		dataspace: dataspace,
		datatype:  datatype,
		layout:    nil,
	}

	return ds, nil
}

// CreateDatasetWithType creates a new dataset with explicit dimensions and datatype.
func (g *Group) CreateDatasetWithType(name string, dims []uint64, dt *message.Datatype, opts ...DatasetOption) (*Dataset, error) {
	if !g.file.writable {
		return nil, fmt.Errorf("file is not writable")
	}

	if name == "" {
		return nil, fmt.Errorf("dataset name cannot be empty")
	}

	options := defaultDatasetOptions()
	for _, opt := range opts {
		if opt != nil {
			opt(options)
		}
	}

	// Create dataspace
	dataspace := message.NewDataspace(dims, options.maxDims)

	// Calculate total size
	numElements := uint64(1)
	for _, d := range dims {
		numElements *= d
	}
	dataSize := dtype.DataSize(dt, numElements)

	// Allocate space for data (will be written later)
	dataAddr := g.file.allocate(int64(dataSize))

	// Create layout
	layout := message.NewContiguousLayout(dataAddr, dataSize)

	// Create dataset object header
	messages := object.NewDatasetHeader(dataspace, dt, layout)

	// Calculate header size and allocate
	headerSize := object.HeaderSize(g.file.writer, messages)
	datasetAddr := g.file.allocate(int64(headerSize))

	// Write the dataset object header
	hw := g.file.writer.At(int64(datasetAddr))
	if _, err := object.WriteHeader(hw, messages); err != nil {
		return nil, fmt.Errorf("writing dataset header: %w", err)
	}

	// Create a hard link from parent group to this dataset
	link := message.NewHardLink(name, datasetAddr)
	if err := g.addLink(link); err != nil {
		return nil, fmt.Errorf("adding link to parent: %w", err)
	}

	// Calculate the path
	newPath := path.Join(g.path, name)
	if g.path == "/" {
		newPath = "/" + name
	}

	// Create the Dataset object with write capability
	ds := &Dataset{
		file:      g.file,
		path:      newPath,
		header:    nil,
		dataspace: dataspace,
		datatype:  dt,
		layout:    nil,
		// Write support
		dataAddr:    dataAddr,
		dataSize:    dataSize,
		numElements: numElements,
	}

	return ds, nil
}

// Write writes data to a dataset that was created with CreateDatasetWithType.
func (ds *Dataset) Write(data interface{}) error {
	if !ds.file.writable {
		return fmt.Errorf("file is not writable")
	}

	if ds.dataAddr == 0 {
		return fmt.Errorf("dataset was not created for writing")
	}

	// Encode the data
	rawData, err := dtype.Encode(ds.datatype, data)
	if err != nil {
		return fmt.Errorf("encoding data: %w", err)
	}

	// Verify size matches
	if uint64(len(rawData)) != ds.dataSize {
		return fmt.Errorf("data size mismatch: expected %d, got %d", ds.dataSize, len(rawData))
	}

	// Write the raw data
	w := ds.file.writer.At(int64(ds.dataAddr))
	if err := w.WriteBytes(rawData); err != nil {
		return fmt.Errorf("writing data: %w", err)
	}

	return nil
}

// inferDimensionsAndType infers the dimensions and element type from a Go value.
func inferDimensionsAndType(val reflect.Value) ([]uint64, reflect.Type, error) {
	var dims []uint64
	current := val

	// Traverse nested slices/arrays to find dimensions
	for {
		switch current.Kind() {
		case reflect.Slice, reflect.Array:
			dims = append(dims, uint64(current.Len()))
			if current.Len() == 0 {
				// Empty slice - get element type from type
				return dims, current.Type().Elem(), nil
			}
			current = current.Index(0)
		default:
			// Reached the element type
			if len(dims) == 0 {
				// Scalar value
				dims = []uint64{1}
			}
			return dims, current.Type(), nil
		}
	}
}

// createAttributeMessage creates an attribute message from a name and value.
func createAttributeMessage(name string, value interface{}) (*message.Attribute, error) {
	// Get the value and type
	val := reflect.ValueOf(value)
	if val.Kind() == reflect.Ptr {
		val = val.Elem()
	}

	// Check if this is a string type
	if val.Kind() == reflect.String {
		return createStringAttribute(name, val.String())
	}

	// Check if this is a slice of strings
	if val.Kind() == reflect.Slice && val.Type().Elem().Kind() == reflect.String {
		return createStringArrayAttribute(name, val)
	}

	// Determine if scalar or array
	var dims []uint64
	var elemType reflect.Type

	switch val.Kind() {
	case reflect.Slice, reflect.Array:
		dims = []uint64{uint64(val.Len())}
		if val.Len() > 0 {
			elemType = val.Index(0).Type()
		} else {
			elemType = val.Type().Elem()
		}
	default:
		// Scalar
		dims = nil // scalar dataspace
		elemType = val.Type()
	}

	// Create datatype from element type
	datatype, err := dtype.GoTypeToDatatype(elemType)
	if err != nil {
		return nil, fmt.Errorf("unsupported attribute type %v: %w", elemType, err)
	}

	// Create dataspace
	var dataspace *message.Dataspace
	if dims == nil {
		dataspace = message.NewScalarDataspace()
	} else {
		dataspace = message.NewDataspace(dims, nil)
	}

	// Encode the value to bytes
	data, err := dtype.Encode(datatype, value)
	if err != nil {
		return nil, fmt.Errorf("encoding attribute value: %w", err)
	}

	return message.NewAttribute(name, datatype, dataspace, data), nil
}

// createStringAttribute creates an attribute with a fixed-length string value.
func createStringAttribute(name string, s string) (*message.Attribute, error) {
	// Use fixed-length string (add 1 for null terminator)
	strLen := len(s) + 1

	// Create fixed-length string datatype
	datatype := message.NewStringDatatype(uint32(strLen), message.PadNullTerm, message.CharsetASCII)

	// Create scalar dataspace
	dataspace := message.NewScalarDataspace()

	// Encode string with null terminator
	data := make([]byte, strLen)
	copy(data, s)
	data[len(s)] = 0

	return message.NewAttribute(name, datatype, dataspace, data), nil
}

// createStringArrayAttribute creates an attribute with an array of fixed-length strings.
func createStringArrayAttribute(name string, val reflect.Value) (*message.Attribute, error) {
	n := val.Len()
	if n == 0 {
		return nil, fmt.Errorf("empty string array not supported")
	}

	// Find maximum string length
	maxLen := 0
	for i := 0; i < n; i++ {
		s := val.Index(i).String()
		if len(s) > maxLen {
			maxLen = len(s)
		}
	}

	// Add 1 for null terminator
	strLen := maxLen + 1

	// Create fixed-length string datatype
	datatype := message.NewStringDatatype(uint32(strLen), message.PadNullTerm, message.CharsetASCII)

	// Create 1D dataspace
	dataspace := message.NewDataspace([]uint64{uint64(n)}, nil)

	// Encode all strings
	data := make([]byte, n*strLen)
	for i := 0; i < n; i++ {
		s := val.Index(i).String()
		offset := i * strLen
		copy(data[offset:], s)
		data[offset+len(s)] = 0
	}

	return message.NewAttribute(name, datatype, dataspace, data), nil
}
