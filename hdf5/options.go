package hdf5

// FileOption configures file creation options.
type FileOption func(*fileOptions)

type fileOptions struct {
	offsetSize int
	lengthSize int
}

func defaultFileOptions() *fileOptions {
	return &fileOptions{
		offsetSize: 8,
		lengthSize: 8,
	}
}

// WithOffsetSize sets the size in bytes for file offsets.
// Valid values are 2, 4, or 8 bytes. Any other value will cause a panic.
func WithOffsetSize(size int) FileOption {
	if size != 2 && size != 4 && size != 8 {
		panic("WithOffsetSize: size must be 2, 4, or 8")
	}
	return func(o *fileOptions) {
		o.offsetSize = size
	}
}

// WithLengthSize sets the size in bytes for lengths.
// Valid values are 2, 4, or 8 bytes. Any other value will cause a panic.
func WithLengthSize(size int) FileOption {
	if size != 2 && size != 4 && size != 8 {
		panic("WithLengthSize: size must be 2, 4, or 8")
	}
	return func(o *fileOptions) {
		o.lengthSize = size
	}
}

// DatasetOption configures dataset creation options.
type DatasetOption func(*datasetOptions)

// attrDef holds an attribute definition for creation.
type attrDef struct {
	name  string
	value interface{}
}

type datasetOptions struct {
	chunks         []uint64
	maxDims        []uint64
	compressionLvl int
	shuffle        bool
	fletcher32     bool
	attributes     []attrDef
}

func defaultDatasetOptions() *datasetOptions {
	return &datasetOptions{
		compressionLvl: 0,
		shuffle:        false,
		fletcher32:     false,
	}
}

// WithChunks sets the chunk dimensions for a chunked dataset.
// Required for resizable datasets and compression.
func WithChunks(dims ...uint64) DatasetOption {
	return func(o *datasetOptions) {
		o.chunks = dims
	}
}

// WithMaxDims sets the maximum dimensions for a resizable dataset.
// Use 0 for unlimited dimension.
func WithMaxDims(dims ...uint64) DatasetOption {
	return func(o *datasetOptions) {
		o.maxDims = dims
	}
}

// WithCompression sets the compression level.
// Valid values are 0 (no compression) through 9 (maximum compression).
// Any value outside this range will cause a panic.
func WithCompression(level int) DatasetOption {
	if level < 0 || level > 9 {
		panic("WithCompression: level must be 0-9")
	}
	return func(o *datasetOptions) {
		o.compressionLvl = level
	}
}

// WithShuffle enables the shuffle filter (improves compression).
func WithShuffle() DatasetOption {
	return func(o *datasetOptions) {
		o.shuffle = true
	}
}

// WithFletcher32 enables Fletcher32 checksum validation.
func WithFletcher32() DatasetOption {
	return func(o *datasetOptions) {
		o.fletcher32 = true
	}
}

// WithAttribute adds an attribute to the dataset.
// The value can be a scalar or slice of: int, int8-64, uint, uint8-64, float32, float64, string.
// Multiple WithAttribute options can be used to add multiple attributes.
func WithAttribute(name string, value interface{}) DatasetOption {
	return func(o *datasetOptions) {
		o.attributes = append(o.attributes, attrDef{name: name, value: value})
	}
}
