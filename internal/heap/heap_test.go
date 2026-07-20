package heap

import (
	"bytes"
	"testing"

	"github.com/huangzhengshun/go-hdf5/internal/binary"
)

// TestLocalHeapGetString tests the LocalHeap.GetString method
func TestLocalHeapGetString(t *testing.T) {
	// Create a local heap with known data
	heap := &LocalHeap{
		DataSize:    20,
		FreeOffset:  20,
		DataAddress: 0,
		data:        []byte("hello\x00world\x00test\x00\x00\x00"),
	}

	tests := []struct {
		name   string
		offset uint64
		want   string
	}{
		{"first string", 0, "hello"},
		{"second string", 6, "world"},
		{"third string", 12, "test"},
		{"empty at end", 17, ""},
		{"out of bounds", 100, ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := heap.GetString(tt.offset)
			if got != tt.want {
				t.Errorf("GetString(%d) = %q, want %q", tt.offset, got, tt.want)
			}
		})
	}
}

func TestLocalHeapGetStringEmpty(t *testing.T) {
	heap := &LocalHeap{
		data: []byte{},
	}

	got := heap.GetString(0)
	if got != "" {
		t.Errorf("expected empty string for empty heap, got %q", got)
	}
}

func TestLocalHeapGetStringNoNullTerminator(t *testing.T) {
	// String that fills entire buffer without null terminator
	heap := &LocalHeap{
		data: []byte("noterm"),
	}

	got := heap.GetString(0)
	if got != "noterm" {
		t.Errorf("expected 'noterm', got %q", got)
	}
}

func TestReadLocalHeapInvalidSignature(t *testing.T) {
	buf := bytes.NewBuffer(nil)
	buf.WriteString("XXXX") // Invalid signature

	r := binary.NewReader(bytes.NewReader(buf.Bytes()), binary.Config{
		OffsetSize: 8,
		LengthSize: 8,
	})

	_, err := ReadLocalHeap(r, 0)
	if err == nil {
		t.Error("expected error for invalid signature")
	}
	if !bytes.Contains([]byte(err.Error()), []byte("invalid local heap signature")) {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestReadLocalHeapUnsupportedVersion(t *testing.T) {
	buf := bytes.NewBuffer(nil)
	buf.WriteString("HEAP") // Valid signature
	buf.WriteByte(5)        // Unsupported version (not 0)

	r := binary.NewReader(bytes.NewReader(buf.Bytes()), binary.Config{
		OffsetSize: 8,
		LengthSize: 8,
	})

	_, err := ReadLocalHeap(r, 0)
	if err == nil {
		t.Error("expected error for unsupported version")
	}
	if !bytes.Contains([]byte(err.Error()), []byte("unsupported local heap version")) {
		t.Errorf("unexpected error: %v", err)
	}
}

// TestGlobalHeapGetObject tests the GlobalHeap.GetObject method
func TestGlobalHeapGetObject(t *testing.T) {
	heap := &GlobalHeap{
		CollectionSize: 100,
		objects: map[uint16][]byte{
			1: []byte("first object"),
			2: []byte{0x01, 0x02, 0x03, 0x04},
			3: []byte(""),
		},
	}

	tests := []struct {
		name    string
		index   uint16
		wantLen int
		wantErr bool
	}{
		{"existing object 1", 1, 12, false},
		{"existing object 2", 2, 4, false},
		{"empty object", 3, 0, false},
		{"non-existent", 99, 0, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := heap.GetObject(tt.index)
			if tt.wantErr {
				if err == nil {
					t.Error("expected error, got nil")
				}
			} else {
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				if len(data) != tt.wantLen {
					t.Errorf("got len %d, want %d", len(data), tt.wantLen)
				}
			}
		})
	}
}

func TestGlobalHeapGetObjectNilHeap(t *testing.T) {
	var heap *GlobalHeap
	_, err := heap.GetObject(1)
	if err == nil {
		t.Error("expected error for nil heap")
	}
}

func TestGlobalHeapGetObjectReturnsCopy(t *testing.T) {
	original := []byte{1, 2, 3, 4}
	heap := &GlobalHeap{
		objects: map[uint16][]byte{
			1: original,
		},
	}

	data, err := heap.GetObject(1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Modify the returned data
	data[0] = 99

	// Original should be unchanged
	if original[0] != 1 {
		t.Error("GetObject should return a copy, not the original slice")
	}
}

// TestGlobalHeapGetString tests the GlobalHeap.GetString method
func TestGlobalHeapGetString(t *testing.T) {
	heap := &GlobalHeap{
		objects: map[uint16][]byte{
			1: []byte("hello\x00"),
			2: []byte("world"),     // No null terminator
			3: []byte{0x00},        // Empty string
			4: []byte("a\x00extra"), // Null in middle
		},
	}

	tests := []struct {
		name    string
		index   uint16
		want    string
		wantErr bool
	}{
		{"with null terminator", 1, "hello", false},
		{"without null terminator", 2, "world", false},
		{"empty string", 3, "", false},
		{"null in middle", 4, "a", false},
		{"non-existent", 99, "", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := heap.GetString(tt.index)
			if tt.wantErr {
				if err == nil {
					t.Error("expected error, got nil")
				}
			} else {
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				if got != tt.want {
					t.Errorf("got %q, want %q", got, tt.want)
				}
			}
		})
	}
}

// TestParseGlobalHeapID tests parsing global heap IDs
func TestParseGlobalHeapID(t *testing.T) {
	tests := []struct {
		name       string
		data       []byte
		offsetSize int
		wantAddr   uint64
		wantIndex  uint32
		wantErr    bool
	}{
		{
			name:       "8-byte offset",
			data:       []byte{0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00},
			offsetSize: 8,
			wantAddr:   0x1000,
			wantIndex:  1,
			wantErr:    false,
		},
		{
			name:       "4-byte offset",
			data:       []byte{0x00, 0x20, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00},
			offsetSize: 4,
			wantAddr:   0x2000,
			wantIndex:  2,
			wantErr:    false,
		},
		{
			name:       "2-byte offset",
			data:       []byte{0x00, 0x30, 0x03, 0x00, 0x00, 0x00},
			offsetSize: 2,
			wantAddr:   0x3000,
			wantIndex:  3,
			wantErr:    false,
		},
		{
			name:       "too short",
			data:       []byte{0x00, 0x00},
			offsetSize: 8,
			wantErr:    true,
		},
		{
			name:       "unsupported offset size",
			data:       []byte{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
			offsetSize: 3,
			wantErr:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			id, err := ParseGlobalHeapID(tt.data, tt.offsetSize)
			if tt.wantErr {
				if err == nil {
					t.Error("expected error, got nil")
				}
			} else {
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				if id.CollectionAddress != tt.wantAddr {
					t.Errorf("address: got 0x%x, want 0x%x", id.CollectionAddress, tt.wantAddr)
				}
				if id.ObjectIndex != tt.wantIndex {
					t.Errorf("index: got %d, want %d", id.ObjectIndex, tt.wantIndex)
				}
			}
		})
	}
}

func TestReadGlobalHeapInvalidAddress(t *testing.T) {
	r := binary.NewReader(bytes.NewReader([]byte{}), binary.Config{
		OffsetSize: 8,
		LengthSize: 8,
	})

	// Test with address 0
	_, err := ReadGlobalHeap(r, 0)
	if err == nil {
		t.Error("expected error for address 0")
	}

	// Test with undefined address
	_, err = ReadGlobalHeap(r, 0xFFFFFFFFFFFFFFFF)
	if err == nil {
		t.Error("expected error for undefined address")
	}
}

func TestReadGlobalHeapInvalidSignature(t *testing.T) {
	buf := bytes.NewBuffer(nil)
	buf.WriteString("XXXX") // Invalid signature

	r := binary.NewReader(bytes.NewReader(buf.Bytes()), binary.Config{
		OffsetSize: 8,
		LengthSize: 8,
	})

	_, err := ReadGlobalHeap(r, 1) // Non-zero address
	if err == nil {
		t.Error("expected error for invalid signature")
	}
}

func TestReadGlobalHeapUnsupportedVersion(t *testing.T) {
	buf := bytes.NewBuffer(nil)
	buf.WriteString("GCOL") // Valid signature
	buf.WriteByte(2)        // Unsupported version

	r := binary.NewReader(bytes.NewReader(buf.Bytes()), binary.Config{
		OffsetSize: 8,
		LengthSize: 8,
	})

	_, err := ReadGlobalHeap(r, 1)
	if err == nil {
		t.Error("expected error for unsupported version")
	}
}

func TestGlobalHeapIDStruct(t *testing.T) {
	id := GlobalHeapID{
		CollectionAddress: 0x1234,
		ObjectIndex:       42,
	}

	if id.CollectionAddress != 0x1234 {
		t.Errorf("unexpected address: 0x%x", id.CollectionAddress)
	}
	if id.ObjectIndex != 42 {
		t.Errorf("unexpected index: %d", id.ObjectIndex)
	}
}
