package filter

import (
	"bytes"
	"compress/zlib"
	"testing"

	"github.com/huangzhengshun/go-hdf5/internal/binary"
	"github.com/huangzhengshun/go-hdf5/internal/message"
)

func TestDeflateRoundtrip(t *testing.T) {
	original := []byte("Hello, World! This is test data for compression testing.")

	// Compress with zlib
	var buf bytes.Buffer
	w := zlib.NewWriter(&buf)
	w.Write(original)
	w.Close()
	compressed := buf.Bytes()

	// Decompress with our filter
	f := NewDeflate(nil)
	decompressed, err := f.Decode(compressed)
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}

	if !bytes.Equal(decompressed, original) {
		t.Errorf("Decompressed data mismatch:\ngot:  %q\nwant: %q", decompressed, original)
	}
}

func TestDeflateID(t *testing.T) {
	f := NewDeflate(nil)
	if f.ID() != message.FilterDeflate {
		t.Errorf("expected ID %d, got %d", message.FilterDeflate, f.ID())
	}
}

func TestShuffleUnshuffle(t *testing.T) {
	// Test data: 4 elements of 4 bytes each
	// Original: [A0 A1 A2 A3] [B0 B1 B2 B3] [C0 C1 C2 C3] [D0 D1 D2 D3]
	// Shuffled: [A0 B0 C0 D0] [A1 B1 C1 D1] [A2 B2 C2 D2] [A3 B3 C3 D3]
	original := []byte{
		0x01, 0x02, 0x03, 0x04, // Element 0
		0x11, 0x12, 0x13, 0x14, // Element 1
		0x21, 0x22, 0x23, 0x24, // Element 2
		0x31, 0x32, 0x33, 0x34, // Element 3
	}

	// Manually shuffle
	shuffled := []byte{
		0x01, 0x11, 0x21, 0x31, // All byte 0s
		0x02, 0x12, 0x22, 0x32, // All byte 1s
		0x03, 0x13, 0x23, 0x33, // All byte 2s
		0x04, 0x14, 0x24, 0x34, // All byte 3s
	}

	f := NewShuffle([]uint32{4}) // 4-byte elements
	unshuffled, err := f.Decode(shuffled)
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}

	if !bytes.Equal(unshuffled, original) {
		t.Errorf("Unshuffled data mismatch:\ngot:  %v\nwant: %v", unshuffled, original)
	}
}

func TestShuffleSingleByte(t *testing.T) {
	// Single-byte elements should pass through unchanged
	data := []byte{1, 2, 3, 4, 5}
	f := NewShuffle([]uint32{1})

	result, err := f.Decode(data)
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}

	if !bytes.Equal(result, data) {
		t.Errorf("Single-byte shuffle should be identity")
	}
}

func TestShuffleID(t *testing.T) {
	f := NewShuffle(nil)
	if f.ID() != message.FilterShuffle {
		t.Errorf("expected ID %d, got %d", message.FilterShuffle, f.ID())
	}
}

func TestFletcher32Valid(t *testing.T) {
	data := []byte("test data for checksum")
	checksum := binary.Fletcher32(data)

	// Append checksum (little-endian)
	input := make([]byte, len(data)+4)
	copy(input, data)
	input[len(data)] = byte(checksum)
	input[len(data)+1] = byte(checksum >> 8)
	input[len(data)+2] = byte(checksum >> 16)
	input[len(data)+3] = byte(checksum >> 24)

	f := NewFletcher32(nil)
	output, err := f.Decode(input)
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}

	if !bytes.Equal(output, data) {
		t.Errorf("Output mismatch:\ngot:  %v\nwant: %v", output, data)
	}
}

func TestFletcher32Invalid(t *testing.T) {
	data := []byte("test data for checksum")

	// Append wrong checksum
	input := make([]byte, len(data)+4)
	copy(input, data)
	input[len(data)] = 0xDE
	input[len(data)+1] = 0xAD
	input[len(data)+2] = 0xBE
	input[len(data)+3] = 0xEF

	f := NewFletcher32(nil)
	_, err := f.Decode(input)
	if err == nil {
		t.Error("Expected error for invalid checksum")
	}
}

func TestFletcher32ID(t *testing.T) {
	f := NewFletcher32(nil)
	if f.ID() != message.FilterFletcher32 {
		t.Errorf("expected ID %d, got %d", message.FilterFletcher32, f.ID())
	}
}

func TestPipelineEmpty(t *testing.T) {
	p, err := NewPipeline(nil)
	if err != nil {
		t.Fatalf("NewPipeline failed: %v", err)
	}

	if !p.Empty() {
		t.Error("Expected empty pipeline")
	}

	data := []byte("unchanged")
	result, err := p.Decode(data, 0)
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}

	if !bytes.Equal(result, data) {
		t.Error("Empty pipeline should pass data through unchanged")
	}
}

func TestPipelineWithFilters(t *testing.T) {
	// Create a pipeline with shuffle + deflate
	fp := &message.FilterPipeline{
		Version: 2,
		Filters: []message.FilterInfo{
			{ID: message.FilterShuffle, ClientData: []uint32{4}},
			{ID: message.FilterDeflate, ClientData: []uint32{6}},
		},
	}

	p, err := NewPipeline(fp)
	if err != nil {
		t.Fatalf("NewPipeline failed: %v", err)
	}

	if p.Len() != 2 {
		t.Errorf("expected 2 filters, got %d", p.Len())
	}
}

func TestPipelineFilterMask(t *testing.T) {
	// Test that filter mask correctly skips filters
	fp := &message.FilterPipeline{
		Version: 2,
		Filters: []message.FilterInfo{
			{ID: message.FilterShuffle, ClientData: []uint32{1}}, // Will be skipped
		},
	}

	p, err := NewPipeline(fp)
	if err != nil {
		t.Fatalf("NewPipeline failed: %v", err)
	}

	data := []byte{1, 2, 3, 4}

	// Filter mask bit 0 set = skip filter 0
	result, err := p.Decode(data, 0x01)
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}

	// Data should be unchanged since shuffle was skipped
	if !bytes.Equal(result, data) {
		t.Error("Skipped filter should leave data unchanged")
	}
}
