package binary

import (
	"bytes"
	"encoding/binary"
	"testing"
)

// bytesReaderAt wraps a byte slice to implement io.ReaderAt.
type bytesReaderAt []byte

func (b bytesReaderAt) ReadAt(p []byte, off int64) (int, error) {
	if off >= int64(len(b)) {
		return 0, nil
	}
	n := copy(p, b[off:])
	return n, nil
}

func TestReaderReadUint8(t *testing.T) {
	data := bytesReaderAt{0x42, 0xFF, 0x00}
	r := NewReader(data, DefaultConfig())

	v, err := r.ReadUint8()
	if err != nil {
		t.Fatalf("ReadUint8 failed: %v", err)
	}
	if v != 0x42 {
		t.Errorf("expected 0x42, got 0x%02x", v)
	}

	v, err = r.ReadUint8()
	if err != nil {
		t.Fatalf("ReadUint8 failed: %v", err)
	}
	if v != 0xFF {
		t.Errorf("expected 0xFF, got 0x%02x", v)
	}
}

func TestReaderReadUint16(t *testing.T) {
	// Little-endian: 0x0102 stored as [0x02, 0x01]
	data := bytesReaderAt{0x02, 0x01, 0xFF, 0xFF}
	r := NewReader(data, DefaultConfig())

	v, err := r.ReadUint16()
	if err != nil {
		t.Fatalf("ReadUint16 failed: %v", err)
	}
	if v != 0x0102 {
		t.Errorf("expected 0x0102, got 0x%04x", v)
	}

	v, err = r.ReadUint16()
	if err != nil {
		t.Fatalf("ReadUint16 failed: %v", err)
	}
	if v != 0xFFFF {
		t.Errorf("expected 0xFFFF, got 0x%04x", v)
	}
}

func TestReaderReadUint32(t *testing.T) {
	var buf bytes.Buffer
	binary.Write(&buf, binary.LittleEndian, uint32(0x12345678))
	binary.Write(&buf, binary.LittleEndian, uint32(0xDEADBEEF))

	r := NewReader(bytesReaderAt(buf.Bytes()), DefaultConfig())

	v, err := r.ReadUint32()
	if err != nil {
		t.Fatalf("ReadUint32 failed: %v", err)
	}
	if v != 0x12345678 {
		t.Errorf("expected 0x12345678, got 0x%08x", v)
	}

	v, err = r.ReadUint32()
	if err != nil {
		t.Fatalf("ReadUint32 failed: %v", err)
	}
	if v != 0xDEADBEEF {
		t.Errorf("expected 0xDEADBEEF, got 0x%08x", v)
	}
}

func TestReaderReadUint64(t *testing.T) {
	var buf bytes.Buffer
	binary.Write(&buf, binary.LittleEndian, uint64(0x123456789ABCDEF0))

	r := NewReader(bytesReaderAt(buf.Bytes()), DefaultConfig())

	v, err := r.ReadUint64()
	if err != nil {
		t.Fatalf("ReadUint64 failed: %v", err)
	}
	if v != 0x123456789ABCDEF0 {
		t.Errorf("expected 0x123456789ABCDEF0, got 0x%016x", v)
	}
}

func TestReaderReadOffset(t *testing.T) {
	tests := []struct {
		name       string
		offsetSize int
		data       []byte
		expected   uint64
	}{
		{"2-byte", 2, []byte{0x34, 0x12}, 0x1234},
		{"4-byte", 4, []byte{0x78, 0x56, 0x34, 0x12}, 0x12345678},
		{"8-byte", 8, []byte{0xF0, 0xDE, 0xBC, 0x9A, 0x78, 0x56, 0x34, 0x12}, 0x123456789ABCDEF0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := Config{
				ByteOrder:  binary.LittleEndian,
				OffsetSize: tt.offsetSize,
				LengthSize: tt.offsetSize,
			}
			r := NewReader(bytesReaderAt(tt.data), cfg)

			v, err := r.ReadOffset()
			if err != nil {
				t.Fatalf("ReadOffset failed: %v", err)
			}
			if v != tt.expected {
				t.Errorf("expected 0x%x, got 0x%x", tt.expected, v)
			}
		})
	}
}

func TestReaderAt(t *testing.T) {
	data := bytesReaderAt{0x00, 0x01, 0x02, 0x03, 0x04, 0x05}
	r := NewReader(data, DefaultConfig())

	// Read from offset 3
	r2 := r.At(3)
	v, err := r2.ReadUint8()
	if err != nil {
		t.Fatalf("ReadUint8 failed: %v", err)
	}
	if v != 0x03 {
		t.Errorf("expected 0x03, got 0x%02x", v)
	}

	// Original reader should be unaffected
	v, err = r.ReadUint8()
	if err != nil {
		t.Fatalf("ReadUint8 failed: %v", err)
	}
	if v != 0x00 {
		t.Errorf("expected 0x00, got 0x%02x", v)
	}
}

func TestReaderSkip(t *testing.T) {
	data := bytesReaderAt{0x00, 0x01, 0x02, 0x03, 0x04}
	r := NewReader(data, DefaultConfig())

	r.Skip(2)
	v, err := r.ReadUint8()
	if err != nil {
		t.Fatalf("ReadUint8 failed: %v", err)
	}
	if v != 0x02 {
		t.Errorf("expected 0x02, got 0x%02x", v)
	}
}

func TestReaderAlign(t *testing.T) {
	tests := []struct {
		startPos  int64
		alignment int64
		expected  int64
	}{
		{0, 8, 0},   // Already aligned
		{1, 8, 8},   // Advance to 8
		{7, 8, 8},   // Advance to 8
		{8, 8, 8},   // Already aligned
		{9, 8, 16},  // Advance to 16
		{0, 4, 0},
		{1, 4, 4},
		{3, 4, 4},
		{4, 4, 4},
	}

	for _, tt := range tests {
		data := make(bytesReaderAt, 32)
		r := NewReader(data, DefaultConfig())
		r.Skip(tt.startPos)
		r.Align(tt.alignment)

		if r.Pos() != tt.expected {
			t.Errorf("Align(%d) from pos %d: expected pos %d, got %d",
				tt.alignment, tt.startPos, tt.expected, r.Pos())
		}
	}
}

func TestReaderPeek(t *testing.T) {
	data := bytesReaderAt{0x00, 0x01, 0x02, 0x03}
	r := NewReader(data, DefaultConfig())

	// Peek should not advance position
	peeked, err := r.Peek(2)
	if err != nil {
		t.Fatalf("Peek failed: %v", err)
	}
	if !bytes.Equal(peeked, []byte{0x00, 0x01}) {
		t.Errorf("expected [0x00, 0x01], got %v", peeked)
	}

	if r.Pos() != 0 {
		t.Errorf("Peek should not advance position, got %d", r.Pos())
	}

	// Read should still get the same data
	read, err := r.ReadBytes(2)
	if err != nil {
		t.Fatalf("ReadBytes failed: %v", err)
	}
	if !bytes.Equal(read, peeked) {
		t.Errorf("Read after Peek mismatch: %v vs %v", read, peeked)
	}
}

func TestReaderIsUndefinedOffset(t *testing.T) {
	tests := []struct {
		offsetSize int
		value      uint64
		expected   bool
	}{
		{2, 0xFFFF, true},
		{2, 0xFFFE, false},
		{4, 0xFFFFFFFF, true},
		{4, 0xFFFFFFFE, false},
		{8, 0xFFFFFFFFFFFFFFFF, true},
		{8, 0xFFFFFFFFFFFFFFFE, false},
	}

	for _, tt := range tests {
		cfg := Config{
			ByteOrder:  binary.LittleEndian,
			OffsetSize: tt.offsetSize,
			LengthSize: tt.offsetSize,
		}
		r := NewReader(bytesReaderAt{}, cfg)

		result := r.IsUndefinedOffset(tt.value)
		if result != tt.expected {
			t.Errorf("IsUndefinedOffset(%d, 0x%x): expected %v, got %v",
				tt.offsetSize, tt.value, tt.expected, result)
		}
	}
}

func TestReaderWithSizes(t *testing.T) {
	data := bytesReaderAt{0x34, 0x12, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}
	r := NewReader(data, DefaultConfig())

	// Default is 8-byte offsets
	r2 := r.WithSizes(2, 2)

	v, err := r2.ReadOffset()
	if err != nil {
		t.Fatalf("ReadOffset failed: %v", err)
	}
	if v != 0x1234 {
		t.Errorf("expected 0x1234, got 0x%x", v)
	}
}
