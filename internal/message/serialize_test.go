package message

import (
	"bytes"
	"io"
	"testing"

	binpkg "github.com/huangzhengshun/go-hdf5/internal/binary"
)

// bytesWriterAt implements io.WriterAt for testing
type bytesWriterAt struct {
	buf []byte
}

func newBytesWriterAt(size int) *bytesWriterAt {
	return &bytesWriterAt{buf: make([]byte, size)}
}

func (b *bytesWriterAt) WriteAt(p []byte, off int64) (n int, err error) {
	if off < 0 {
		return 0, io.ErrUnexpectedEOF
	}
	if int(off)+len(p) > len(b.buf) {
		newBuf := make([]byte, int(off)+len(p))
		copy(newBuf, b.buf)
		b.buf = newBuf
	}
	copy(b.buf[off:], p)
	return len(p), nil
}

func (b *bytesWriterAt) Bytes() []byte {
	return b.buf
}

func TestDataspaceSerialize(t *testing.T) {
	tests := []struct {
		name       string
		dataspace  *Dataspace
		expectSize int
	}{
		{
			name:       "scalar",
			dataspace:  NewScalarDataspace(),
			expectSize: 4, // version + rank + flags + type
		},
		{
			name:       "null",
			dataspace:  NewNullDataspace(),
			expectSize: 4,
		},
		{
			name:       "1D simple",
			dataspace:  NewDataspace([]uint64{100}, nil),
			expectSize: 4 + 8, // header + 1 dimension
		},
		{
			name:       "2D simple",
			dataspace:  NewDataspace([]uint64{10, 20}, nil),
			expectSize: 4 + 16, // header + 2 dimensions
		},
		{
			name:       "with max dims",
			dataspace:  NewDataspace([]uint64{10}, []uint64{100}),
			expectSize: 4 + 8 + 8, // header + dims + maxdims
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			buf := newBytesWriterAt(256)
			cfg := binpkg.DefaultConfig()
			w := binpkg.NewWriter(buf, cfg)

			err := tt.dataspace.Serialize(w)
			if err != nil {
				t.Fatalf("Serialize failed: %v", err)
			}

			if int(w.Pos()) != tt.expectSize {
				t.Errorf("expected size %d, got %d", tt.expectSize, w.Pos())
			}

			// Verify round-trip by parsing
			r := binpkg.NewReader(bytes.NewReader(buf.Bytes()), cfg)
			parsed, err := parseDataspace(buf.Bytes()[:w.Pos()], r)
			if err != nil {
				t.Fatalf("parseDataspace failed: %v", err)
			}

			if parsed.Version != 2 {
				t.Errorf("expected version 2, got %d", parsed.Version)
			}
			if parsed.Rank != tt.dataspace.Rank {
				t.Errorf("expected rank %d, got %d", tt.dataspace.Rank, parsed.Rank)
			}
			if parsed.SpaceType != tt.dataspace.SpaceType {
				t.Errorf("expected type %d, got %d", tt.dataspace.SpaceType, parsed.SpaceType)
			}
		})
	}
}

func TestLinkSerialize(t *testing.T) {
	buf := newBytesWriterAt(256)
	cfg := binpkg.DefaultConfig()
	w := binpkg.NewWriter(buf, cfg)

	link := NewHardLink("test_dataset", 0x1234)
	err := link.Serialize(w)
	if err != nil {
		t.Fatalf("Serialize failed: %v", err)
	}

	// Expected: version(1) + flags(1) + nameLen(1) + name(12) + address(8) = 23
	expectedSize := 1 + 1 + 1 + len("test_dataset") + 8
	if int(w.Pos()) != expectedSize {
		t.Errorf("expected size %d, got %d", expectedSize, w.Pos())
	}

	// Verify round-trip
	r := binpkg.NewReader(bytes.NewReader(buf.Bytes()), cfg)
	parsed, err := parseLink(buf.Bytes()[:w.Pos()], r)
	if err != nil {
		t.Fatalf("parseLink failed: %v", err)
	}

	if parsed.Name != link.Name {
		t.Errorf("expected name %q, got %q", link.Name, parsed.Name)
	}
	if parsed.LinkType != LinkTypeHard {
		t.Errorf("expected hard link, got %d", parsed.LinkType)
	}
	if parsed.ObjectAddress != link.ObjectAddress {
		t.Errorf("expected address 0x%x, got 0x%x", link.ObjectAddress, parsed.ObjectAddress)
	}
}

func TestSoftLinkSerialize(t *testing.T) {
	buf := newBytesWriterAt(256)
	cfg := binpkg.DefaultConfig()
	w := binpkg.NewWriter(buf, cfg)

	link := NewSoftLink("link_name", "/target/path")
	err := link.Serialize(w)
	if err != nil {
		t.Fatalf("Serialize failed: %v", err)
	}

	// Verify round-trip
	r := binpkg.NewReader(bytes.NewReader(buf.Bytes()), cfg)
	parsed, err := parseLink(buf.Bytes()[:w.Pos()], r)
	if err != nil {
		t.Fatalf("parseLink failed: %v", err)
	}

	if parsed.Name != link.Name {
		t.Errorf("expected name %q, got %q", link.Name, parsed.Name)
	}
	if parsed.LinkType != LinkTypeSoft {
		t.Errorf("expected soft link, got %d", parsed.LinkType)
	}
	if parsed.SoftLinkValue != link.SoftLinkValue {
		t.Errorf("expected target %q, got %q", link.SoftLinkValue, parsed.SoftLinkValue)
	}
}

func TestDatatypeSerialize(t *testing.T) {
	tests := []struct {
		name     string
		datatype *Datatype
	}{
		{
			name:     "int32",
			datatype: NewFixedPointDatatype(4, true, OrderLE),
		},
		{
			name:     "uint64",
			datatype: NewFixedPointDatatype(8, false, OrderLE),
		},
		{
			name:     "float32",
			datatype: NewFloatDatatype(4, OrderLE),
		},
		{
			name:     "float64",
			datatype: NewFloatDatatype(8, OrderLE),
		},
		{
			name:     "fixed string",
			datatype: NewStringDatatype(16, PadNullTerm, CharsetUTF8),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			buf := newBytesWriterAt(256)
			cfg := binpkg.DefaultConfig()
			w := binpkg.NewWriter(buf, cfg)

			err := tt.datatype.Serialize(w)
			if err != nil {
				t.Fatalf("Serialize failed: %v", err)
			}

			// Verify round-trip
			r := binpkg.NewReader(bytes.NewReader(buf.Bytes()), cfg)
			parsed, err := parseDatatype(buf.Bytes()[:w.Pos()], r)
			if err != nil {
				t.Fatalf("parseDatatype failed: %v", err)
			}

			if parsed.Class != tt.datatype.Class {
				t.Errorf("expected class %d, got %d", tt.datatype.Class, parsed.Class)
			}
			if parsed.Size != tt.datatype.Size {
				t.Errorf("expected size %d, got %d", tt.datatype.Size, parsed.Size)
			}
		})
	}
}

func TestLayoutSerializeContiguous(t *testing.T) {
	buf := newBytesWriterAt(256)
	cfg := binpkg.DefaultConfig()
	w := binpkg.NewWriter(buf, cfg)

	layout := NewContiguousLayout(0x1000, 1024)
	err := layout.Serialize(w)
	if err != nil {
		t.Fatalf("Serialize failed: %v", err)
	}

	// Expected: version(1) + class(1) + address(8) + size(8) = 18
	expectedSize := 2 + 8 + 8
	if int(w.Pos()) != expectedSize {
		t.Errorf("expected size %d, got %d", expectedSize, w.Pos())
	}

	// Verify round-trip
	r := binpkg.NewReader(bytes.NewReader(buf.Bytes()), cfg)
	parsed, err := parseDataLayout(buf.Bytes()[:w.Pos()], r)
	if err != nil {
		t.Fatalf("parseDataLayout failed: %v", err)
	}

	if parsed.Class != LayoutContiguous {
		t.Errorf("expected contiguous layout, got %d", parsed.Class)
	}
	if parsed.Address != layout.Address {
		t.Errorf("expected address 0x%x, got 0x%x", layout.Address, parsed.Address)
	}
	if parsed.Size != layout.Size {
		t.Errorf("expected size %d, got %d", layout.Size, parsed.Size)
	}
}

func TestLayoutSerializeCompact(t *testing.T) {
	buf := newBytesWriterAt(256)
	cfg := binpkg.DefaultConfig()
	w := binpkg.NewWriter(buf, cfg)

	data := []byte{1, 2, 3, 4, 5, 6, 7, 8}
	layout := NewCompactLayout(data)
	err := layout.Serialize(w)
	if err != nil {
		t.Fatalf("Serialize failed: %v", err)
	}

	// Verify round-trip
	r := binpkg.NewReader(bytes.NewReader(buf.Bytes()), cfg)
	parsed, err := parseDataLayout(buf.Bytes()[:w.Pos()], r)
	if err != nil {
		t.Fatalf("parseDataLayout failed: %v", err)
	}

	if parsed.Class != LayoutCompact {
		t.Errorf("expected compact layout, got %d", parsed.Class)
	}
	if !bytes.Equal(parsed.CompactData, data) {
		t.Errorf("expected data %v, got %v", data, parsed.CompactData)
	}
}

func TestSerializedSize(t *testing.T) {
	cfg := binpkg.DefaultConfig()
	w := binpkg.NewWriter(newBytesWriterAt(256), cfg)

	tests := []struct {
		name string
		msg  Serializable
	}{
		{"scalar dataspace", NewScalarDataspace()},
		{"1D dataspace", NewDataspace([]uint64{100}, nil)},
		{"hard link", NewHardLink("test", 0x1234)},
		{"soft link", NewSoftLink("link", "/path")},
		{"int32 dtype", NewFixedPointDatatype(4, true, OrderLE)},
		{"contiguous layout", NewContiguousLayout(0x1000, 1024)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Calculate predicted size
			predicted := tt.msg.SerializedSize(w)

			// Actually serialize
			buf := newBytesWriterAt(256)
			w2 := binpkg.NewWriter(buf, cfg)
			err := tt.msg.Serialize(w2)
			if err != nil {
				t.Fatalf("Serialize failed: %v", err)
			}

			actual := int(w2.Pos())
			if predicted != actual {
				t.Errorf("SerializedSize predicted %d, actual %d", predicted, actual)
			}
		})
	}
}
