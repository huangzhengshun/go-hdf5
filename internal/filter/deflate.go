package filter

import (
	"bytes"
	"compress/zlib"
	"fmt"
	"io"

	"github.com/huangzhengshun/go-hdf5/internal/message"
)

// Deflate implements the DEFLATE filter (gzip/zlib compression).
type Deflate struct {
	level int
}

// NewDeflate creates a new DEFLATE filter.
// Client data: [0] = compression level (0-9, or default if empty)
func NewDeflate(clientData []uint32) *Deflate {
	level := 6 // Default compression level
	if len(clientData) > 0 {
		level = int(clientData[0])
	}
	return &Deflate{level: level}
}

func (f *Deflate) ID() uint16 {
	return message.FilterDeflate
}

func (f *Deflate) Decode(input []byte) ([]byte, error) {
	r, err := zlib.NewReader(bytes.NewReader(input))
	if err != nil {
		return nil, fmt.Errorf("zlib reader: %w", err)
	}
	defer r.Close()

	output, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("zlib decompress: %w", err)
	}

	return output, nil
}
