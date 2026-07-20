package binary

import (
	"bytes"
	"encoding/binary"
	"io"
	"testing"
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
		// Extend buffer if needed
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

func TestNewWriter(t *testing.T) {
	buf := newBytesWriterAt(64)
	cfg := Config{
		ByteOrder:  binary.LittleEndian,
		OffsetSize: 8,
		LengthSize: 8,
	}
	w := NewWriter(buf, cfg)

	if w.Pos() != 0 {
		t.Errorf("expected initial position 0, got %d", w.Pos())
	}
	if w.OffsetSize() != 8 {
		t.Errorf("expected offset size 8, got %d", w.OffsetSize())
	}
	if w.LengthSize() != 8 {
		t.Errorf("expected length size 8, got %d", w.LengthSize())
	}
}

func TestWriterAt(t *testing.T) {
	buf := newBytesWriterAt(64)
	w := NewWriter(buf, DefaultConfig())

	w2 := w.At(32)
	if w2.Pos() != 32 {
		t.Errorf("expected position 32, got %d", w2.Pos())
	}
	// Original writer should be unchanged
	if w.Pos() != 0 {
		t.Errorf("expected original position 0, got %d", w.Pos())
	}
}

func TestWriterWithSizes(t *testing.T) {
	buf := newBytesWriterAt(64)
	w := NewWriter(buf, DefaultConfig())

	w2 := w.WithSizes(4, 4)
	if w2.OffsetSize() != 4 {
		t.Errorf("expected offset size 4, got %d", w2.OffsetSize())
	}
	if w2.LengthSize() != 4 {
		t.Errorf("expected length size 4, got %d", w2.LengthSize())
	}
	// Original writer should be unchanged
	if w.OffsetSize() != 8 {
		t.Errorf("expected original offset size 8, got %d", w.OffsetSize())
	}
}

func TestWriteBytes(t *testing.T) {
	buf := newBytesWriterAt(64)
	w := NewWriter(buf, DefaultConfig())

	data := []byte{0x01, 0x02, 0x03, 0x04}
	err := w.WriteBytes(data)
	if err != nil {
		t.Fatalf("WriteBytes failed: %v", err)
	}

	if w.Pos() != 4 {
		t.Errorf("expected position 4, got %d", w.Pos())
	}

	if !bytes.Equal(buf.Bytes()[:4], data) {
		t.Errorf("expected %v, got %v", data, buf.Bytes()[:4])
	}
}

func TestWriteUint8(t *testing.T) {
	buf := newBytesWriterAt(64)
	w := NewWriter(buf, DefaultConfig())

	err := w.WriteUint8(0xAB)
	if err != nil {
		t.Fatalf("WriteUint8 failed: %v", err)
	}

	if buf.Bytes()[0] != 0xAB {
		t.Errorf("expected 0xAB, got 0x%02X", buf.Bytes()[0])
	}
}

func TestWriteUint16(t *testing.T) {
	buf := newBytesWriterAt(64)
	w := NewWriter(buf, DefaultConfig())

	err := w.WriteUint16(0x1234)
	if err != nil {
		t.Fatalf("WriteUint16 failed: %v", err)
	}

	// Little-endian: low byte first
	if buf.Bytes()[0] != 0x34 || buf.Bytes()[1] != 0x12 {
		t.Errorf("expected [0x34, 0x12], got [0x%02X, 0x%02X]", buf.Bytes()[0], buf.Bytes()[1])
	}
}

func TestWriteUint32(t *testing.T) {
	buf := newBytesWriterAt(64)
	w := NewWriter(buf, DefaultConfig())

	err := w.WriteUint32(0x12345678)
	if err != nil {
		t.Fatalf("WriteUint32 failed: %v", err)
	}

	expected := []byte{0x78, 0x56, 0x34, 0x12}
	if !bytes.Equal(buf.Bytes()[:4], expected) {
		t.Errorf("expected %v, got %v", expected, buf.Bytes()[:4])
	}
}

func TestWriteUint64(t *testing.T) {
	buf := newBytesWriterAt(64)
	w := NewWriter(buf, DefaultConfig())

	err := w.WriteUint64(0x123456789ABCDEF0)
	if err != nil {
		t.Fatalf("WriteUint64 failed: %v", err)
	}

	expected := []byte{0xF0, 0xDE, 0xBC, 0x9A, 0x78, 0x56, 0x34, 0x12}
	if !bytes.Equal(buf.Bytes()[:8], expected) {
		t.Errorf("expected %v, got %v", expected, buf.Bytes()[:8])
	}
}

func TestWriteOffset(t *testing.T) {
	tests := []struct {
		name       string
		offsetSize int
		value      uint64
		expected   []byte
	}{
		{"2-byte offset", 2, 0x1234, []byte{0x34, 0x12}},
		{"4-byte offset", 4, 0x12345678, []byte{0x78, 0x56, 0x34, 0x12}},
		{"8-byte offset", 8, 0x123456789ABCDEF0, []byte{0xF0, 0xDE, 0xBC, 0x9A, 0x78, 0x56, 0x34, 0x12}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			buf := newBytesWriterAt(64)
			cfg := Config{
				ByteOrder:  binary.LittleEndian,
				OffsetSize: tt.offsetSize,
				LengthSize: tt.offsetSize,
			}
			w := NewWriter(buf, cfg)

			err := w.WriteOffset(tt.value)
			if err != nil {
				t.Fatalf("WriteOffset failed: %v", err)
			}

			if !bytes.Equal(buf.Bytes()[:tt.offsetSize], tt.expected) {
				t.Errorf("expected %v, got %v", tt.expected, buf.Bytes()[:tt.offsetSize])
			}
		})
	}
}

func TestWriteLength(t *testing.T) {
	buf := newBytesWriterAt(64)
	cfg := Config{
		ByteOrder:  binary.LittleEndian,
		OffsetSize: 8,
		LengthSize: 4,
	}
	w := NewWriter(buf, cfg)

	err := w.WriteLength(0x12345678)
	if err != nil {
		t.Fatalf("WriteLength failed: %v", err)
	}

	expected := []byte{0x78, 0x56, 0x34, 0x12}
	if !bytes.Equal(buf.Bytes()[:4], expected) {
		t.Errorf("expected %v, got %v", expected, buf.Bytes()[:4])
	}

	if w.Pos() != 4 {
		t.Errorf("expected position 4, got %d", w.Pos())
	}
}

func TestUndefinedValues(t *testing.T) {
	tests := []struct {
		size     int
		expected uint64
	}{
		{2, 0xFFFF},
		{4, 0xFFFFFFFF},
		{8, 0xFFFFFFFFFFFFFFFF},
	}

	for _, tt := range tests {
		buf := newBytesWriterAt(64)
		cfg := Config{
			ByteOrder:  binary.LittleEndian,
			OffsetSize: tt.size,
			LengthSize: tt.size,
		}
		w := NewWriter(buf, cfg)

		if w.UndefinedOffset() != tt.expected {
			t.Errorf("size %d: expected undefined offset 0x%X, got 0x%X", tt.size, tt.expected, w.UndefinedOffset())
		}
		if w.UndefinedLength() != tt.expected {
			t.Errorf("size %d: expected undefined length 0x%X, got 0x%X", tt.size, tt.expected, w.UndefinedLength())
		}
	}
}

func TestWriteUndefinedOffset(t *testing.T) {
	buf := newBytesWriterAt(64)
	cfg := Config{
		ByteOrder:  binary.LittleEndian,
		OffsetSize: 4,
		LengthSize: 4,
	}
	w := NewWriter(buf, cfg)

	err := w.WriteUndefinedOffset()
	if err != nil {
		t.Fatalf("WriteUndefinedOffset failed: %v", err)
	}

	expected := []byte{0xFF, 0xFF, 0xFF, 0xFF}
	if !bytes.Equal(buf.Bytes()[:4], expected) {
		t.Errorf("expected %v, got %v", expected, buf.Bytes()[:4])
	}
}

func TestSkip(t *testing.T) {
	buf := newBytesWriterAt(64)
	w := NewWriter(buf, DefaultConfig())

	w.Skip(10)
	if w.Pos() != 10 {
		t.Errorf("expected position 10, got %d", w.Pos())
	}

	w.Skip(5)
	if w.Pos() != 15 {
		t.Errorf("expected position 15, got %d", w.Pos())
	}
}

func TestAlign(t *testing.T) {
	tests := []struct {
		startPos  int64
		alignment int64
		expected  int64
	}{
		{0, 8, 0},   // Already aligned
		{1, 8, 8},   // Needs alignment
		{7, 8, 8},   // One byte short
		{8, 8, 8},   // Already aligned
		{9, 8, 16},  // Needs alignment
		{0, 4, 0},   // Already aligned
		{3, 4, 4},   // Needs alignment
		{10, 1, 10}, // Alignment of 1 = no change
		{10, 0, 10}, // Alignment of 0 = no change
	}

	for _, tt := range tests {
		buf := newBytesWriterAt(64)
		w := NewWriter(buf, DefaultConfig())
		w.Skip(tt.startPos)

		w.Align(tt.alignment)
		if w.Pos() != tt.expected {
			t.Errorf("Align(%d) from %d: expected %d, got %d", tt.alignment, tt.startPos, tt.expected, w.Pos())
		}
	}
}

func TestWritePadding(t *testing.T) {
	buf := newBytesWriterAt(64)
	w := NewWriter(buf, DefaultConfig())

	w.Skip(3)
	err := w.WritePadding(8)
	if err != nil {
		t.Fatalf("WritePadding failed: %v", err)
	}

	if w.Pos() != 8 {
		t.Errorf("expected position 8, got %d", w.Pos())
	}

	// Verify zeros were written
	for i := 3; i < 8; i++ {
		if buf.Bytes()[i] != 0 {
			t.Errorf("expected zero at position %d, got 0x%02X", i, buf.Bytes()[i])
		}
	}
}

func TestWriteZeros(t *testing.T) {
	buf := newBytesWriterAt(64)
	w := NewWriter(buf, DefaultConfig())

	// First write some non-zero data
	w.WriteBytes([]byte{0xFF, 0xFF, 0xFF, 0xFF})

	// Reset position
	w = w.At(0)
	err := w.WriteZeros(4)
	if err != nil {
		t.Fatalf("WriteZeros failed: %v", err)
	}

	for i := 0; i < 4; i++ {
		if buf.Bytes()[i] != 0 {
			t.Errorf("expected zero at position %d, got 0x%02X", i, buf.Bytes()[i])
		}
	}
}

func TestWriterRoundTrip(t *testing.T) {
	// Test that what we write can be read back by the Reader
	buf := newBytesWriterAt(64)
	cfg := DefaultConfig()
	w := NewWriter(buf, cfg)

	// Write various values
	w.WriteUint8(0xAB)
	w.WriteUint16(0x1234)
	w.WriteUint32(0xDEADBEEF)
	w.WriteUint64(0x123456789ABCDEF0)
	w.WriteOffset(0xCAFEBABE)

	// Read back with Reader
	r := NewReader(bytes.NewReader(buf.Bytes()), cfg)

	v8, _ := r.ReadUint8()
	if v8 != 0xAB {
		t.Errorf("uint8: expected 0xAB, got 0x%02X", v8)
	}

	v16, _ := r.ReadUint16()
	if v16 != 0x1234 {
		t.Errorf("uint16: expected 0x1234, got 0x%04X", v16)
	}

	v32, _ := r.ReadUint32()
	if v32 != 0xDEADBEEF {
		t.Errorf("uint32: expected 0xDEADBEEF, got 0x%08X", v32)
	}

	v64, _ := r.ReadUint64()
	if v64 != 0x123456789ABCDEF0 {
		t.Errorf("uint64: expected 0x123456789ABCDEF0, got 0x%016X", v64)
	}

	vOff, _ := r.ReadOffset()
	if vOff != 0xCAFEBABE {
		t.Errorf("offset: expected 0xCAFEBABE, got 0x%X", vOff)
	}
}

func TestWriterBigEndian(t *testing.T) {
	buf := newBytesWriterAt(64)
	cfg := Config{
		ByteOrder:  binary.BigEndian,
		OffsetSize: 8,
		LengthSize: 8,
	}
	w := NewWriter(buf, cfg)

	w.WriteUint32(0x12345678)

	// Big-endian: high byte first
	expected := []byte{0x12, 0x34, 0x56, 0x78}
	if !bytes.Equal(buf.Bytes()[:4], expected) {
		t.Errorf("expected %v, got %v", expected, buf.Bytes()[:4])
	}
}

func TestSeekableWriterAt(t *testing.T) {
	// bytes.Buffer doesn't implement Seek, so we just verify the interface compiles
	_ = (*SeekableWriterAt)(nil)
}
