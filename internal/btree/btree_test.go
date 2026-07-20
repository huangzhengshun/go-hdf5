package btree

import (
	"bytes"
	"testing"

	"github.com/huangzhengshun/go-hdf5/internal/binary"
)

func TestChunkIndexFindChunk(t *testing.T) {
	// Create a chunk index with known entries
	idx := &ChunkIndex{
		NDims: 2,
		Entries: []ChunkEntry{
			{Offset: []uint64{0, 0}, FilterMask: 0, Size: 400, Address: 1000},
			{Offset: []uint64{0, 10}, FilterMask: 0, Size: 400, Address: 2000},
			{Offset: []uint64{10, 0}, FilterMask: 0, Size: 400, Address: 3000},
			{Offset: []uint64{10, 10}, FilterMask: 0, Size: 400, Address: 4000},
		},
	}

	chunkDims := []uint32{10, 10}

	tests := []struct {
		name     string
		offset   []uint64
		wantAddr uint64
		wantNil  bool
	}{
		{"first chunk origin", []uint64{0, 0}, 1000, false},
		{"first chunk middle", []uint64{5, 5}, 1000, false},
		{"first chunk edge", []uint64{9, 9}, 1000, false},
		{"second chunk", []uint64{0, 10}, 2000, false},
		{"second chunk middle", []uint64{3, 15}, 2000, false},
		{"third chunk", []uint64{10, 0}, 3000, false},
		{"fourth chunk", []uint64{10, 10}, 4000, false},
		{"fourth chunk edge", []uint64{19, 19}, 4000, false},
		{"out of bounds", []uint64{20, 20}, 0, true},
		{"negative direction (no match)", []uint64{100, 100}, 0, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := idx.FindChunk(tt.offset, chunkDims)
			if tt.wantNil {
				if result != nil {
					t.Errorf("expected nil, got chunk at address %d", result.Address)
				}
			} else {
				if result == nil {
					t.Errorf("expected chunk at address %d, got nil", tt.wantAddr)
				} else if result.Address != tt.wantAddr {
					t.Errorf("expected address %d, got %d", tt.wantAddr, result.Address)
				}
			}
		})
	}
}

func TestChunkIndexFindChunk3D(t *testing.T) {
	// Test 3D chunk finding
	idx := &ChunkIndex{
		NDims: 3,
		Entries: []ChunkEntry{
			{Offset: []uint64{0, 0, 0}, FilterMask: 0, Size: 1000, Address: 5000},
			{Offset: []uint64{0, 0, 8}, FilterMask: 0, Size: 1000, Address: 5100},
			{Offset: []uint64{4, 4, 0}, FilterMask: 1, Size: 800, Address: 5200}, // Filter 0 skipped
		},
	}

	chunkDims := []uint32{4, 4, 8}

	// Test first chunk
	result := idx.FindChunk([]uint64{1, 2, 3}, chunkDims)
	if result == nil || result.Address != 5000 {
		t.Errorf("expected chunk at 5000")
	}

	// Test chunk with filter mask
	result = idx.FindChunk([]uint64{5, 5, 2}, chunkDims)
	if result == nil || result.FilterMask != 1 {
		t.Errorf("expected chunk with FilterMask=1")
	}
}

func TestChunkIndexEmpty(t *testing.T) {
	idx := &ChunkIndex{
		NDims:   2,
		Entries: []ChunkEntry{},
	}

	result := idx.FindChunk([]uint64{0, 0}, []uint32{10, 10})
	if result != nil {
		t.Errorf("expected nil for empty index, got %v", result)
	}
}

func TestReadChunkBTreeInvalidSignature(t *testing.T) {
	// Create a buffer with invalid signature
	buf := bytes.NewBuffer(nil)
	buf.WriteString("XXXX") // Invalid signature

	r := binary.NewReader(bytes.NewReader(buf.Bytes()), binary.Config{
		OffsetSize: 8,
		LengthSize: 8,
	})

	_, err := ReadChunkIndex(r, 0, 2)
	if err == nil {
		t.Error("expected error for invalid signature")
	}
	if !bytes.Contains([]byte(err.Error()), []byte("invalid B-tree signature")) {
		t.Errorf("unexpected error message: %v", err)
	}
}

func TestReadChunkBTreeWrongNodeType(t *testing.T) {
	// Create a buffer with TREE signature but wrong node type (0 = group instead of 1 = chunk)
	buf := bytes.NewBuffer(nil)
	buf.WriteString("TREE")       // Valid signature
	buf.WriteByte(0)              // Node type 0 (group, not chunk)
	buf.WriteByte(0)              // Node level 0 (leaf)
	buf.Write([]byte{0, 0})       // Entries used = 0
	buf.Write(make([]byte, 16))   // Left/right siblings (8 bytes each)

	r := binary.NewReader(bytes.NewReader(buf.Bytes()), binary.Config{
		OffsetSize: 8,
		LengthSize: 8,
	})

	_, err := ReadChunkIndex(r, 0, 2)
	if err == nil {
		t.Error("expected error for wrong node type")
	}
	if !bytes.Contains([]byte(err.Error()), []byte("unexpected B-tree node type")) {
		t.Errorf("unexpected error message: %v", err)
	}
}

func TestGroupEntryStruct(t *testing.T) {
	// Test GroupEntry structure
	entry := GroupEntry{
		Name:          "test_dataset",
		ObjectAddress: 12345,
		LinkType:      0,
		SoftLinkValue: "",
	}

	if entry.Name != "test_dataset" {
		t.Errorf("unexpected name: %s", entry.Name)
	}
	if entry.ObjectAddress != 12345 {
		t.Errorf("unexpected address: %d", entry.ObjectAddress)
	}

	// Test soft link entry
	softEntry := GroupEntry{
		Name:          "soft_link",
		ObjectAddress: 0,
		LinkType:      1,
		SoftLinkValue: "/target/path",
	}

	if softEntry.LinkType != 1 {
		t.Errorf("expected link type 1, got %d", softEntry.LinkType)
	}
	if softEntry.SoftLinkValue != "/target/path" {
		t.Errorf("unexpected soft link value: %s", softEntry.SoftLinkValue)
	}
}

func TestChunkEntryStruct(t *testing.T) {
	entry := ChunkEntry{
		Offset:     []uint64{100, 200, 300},
		FilterMask: 0x3, // Filters 0 and 1 skipped
		Size:       1024,
		Address:    8192,
	}

	if len(entry.Offset) != 3 {
		t.Errorf("expected 3 dimensions, got %d", len(entry.Offset))
	}
	if entry.Offset[0] != 100 || entry.Offset[1] != 200 || entry.Offset[2] != 300 {
		t.Errorf("unexpected offsets: %v", entry.Offset)
	}
	if entry.FilterMask != 0x3 {
		t.Errorf("expected filter mask 3, got %d", entry.FilterMask)
	}
	if entry.Size != 1024 {
		t.Errorf("expected size 1024, got %d", entry.Size)
	}
	if entry.Address != 8192 {
		t.Errorf("expected address 8192, got %d", entry.Address)
	}
}

func TestReadGroupEntriesInvalidSignature(t *testing.T) {
	// Create a buffer with invalid B-tree signature
	buf := bytes.NewBuffer(nil)
	buf.WriteString("XXXX") // Invalid signature

	r := binary.NewReader(bytes.NewReader(buf.Bytes()), binary.Config{
		OffsetSize: 8,
		LengthSize: 8,
	})

	_, err := ReadGroupEntries(r, 0, nil)
	if err == nil {
		t.Error("expected error for invalid signature")
	}
}

// B-tree v2 tests

func TestReadChunkIndexV2InvalidSignature(t *testing.T) {
	// Create a buffer with invalid BTHD signature
	buf := bytes.NewBuffer(nil)
	buf.WriteString("XXXX") // Invalid signature

	r := binary.NewReader(bytes.NewReader(buf.Bytes()), binary.DefaultConfig())

	_, err := ReadChunkIndexV2(r, 0, 2)
	if err == nil {
		t.Error("expected error for invalid signature")
	}
	if !bytes.Contains([]byte(err.Error()), []byte("invalid B-tree v2 signature")) {
		t.Errorf("unexpected error message: %v", err)
	}
}

func TestReadChunkIndexV2WrongVersion(t *testing.T) {
	// Create a buffer with BTHD signature but wrong version
	buf := bytes.NewBuffer(nil)
	buf.WriteString("BTHD") // Valid signature
	buf.WriteByte(1)        // Wrong version (should be 0)

	r := binary.NewReader(bytes.NewReader(buf.Bytes()), binary.DefaultConfig())

	_, err := ReadChunkIndexV2(r, 0, 2)
	if err == nil {
		t.Error("expected error for wrong version")
	}
	if !bytes.Contains([]byte(err.Error()), []byte("unsupported B-tree v2 version")) {
		t.Errorf("unexpected error message: %v", err)
	}
}

func TestReadChunkIndexV2WrongType(t *testing.T) {
	// Create a buffer with BTHD signature but wrong type (not 10 or 11)
	buf := bytes.NewBuffer(nil)
	buf.WriteString("BTHD")             // Valid signature
	buf.WriteByte(0)                    // Version 0
	buf.WriteByte(5)                    // Wrong type (should be 10 or 11)
	buf.Write([]byte{0, 0, 0, 0})       // Node size (4 bytes)
	buf.Write([]byte{0, 0})             // Record size (2 bytes)
	buf.Write([]byte{0, 0})             // Depth (2 bytes)
	buf.WriteByte(0)                    // Split percent
	buf.WriteByte(0)                    // Merge percent
	buf.Write(make([]byte, 8))          // Root address
	buf.Write([]byte{0, 0})             // Num root records
	buf.Write(make([]byte, 8))          // Total records

	r := binary.NewReader(bytes.NewReader(buf.Bytes()), binary.DefaultConfig())

	_, err := ReadChunkIndexV2(r, 0, 2)
	if err == nil {
		t.Error("expected error for wrong type")
	}
	if !bytes.Contains([]byte(err.Error()), []byte("unexpected B-tree v2 type")) {
		t.Errorf("unexpected error message: %v", err)
	}
}

func TestReadChunkIndexV2EmptyIndex(t *testing.T) {
	// Create a valid BTHD with 0 total records
	buf := bytes.NewBuffer(nil)
	buf.WriteString("BTHD")             // Signature
	buf.WriteByte(0)                    // Version 0
	buf.WriteByte(10)                   // Type 10 (chunks without filter)
	buf.Write([]byte{0, 4, 0, 0})       // Node size = 1024
	buf.Write([]byte{24, 0})            // Record size = 24
	buf.Write([]byte{0, 0})             // Depth = 0
	buf.WriteByte(75)                   // Split percent
	buf.WriteByte(25)                   // Merge percent
	buf.Write(make([]byte, 8))          // Root address
	buf.Write([]byte{0, 0})             // Num root records = 0
	buf.Write([]byte{0, 0, 0, 0, 0, 0, 0, 0}) // Total records = 0

	r := binary.NewReader(bytes.NewReader(buf.Bytes()), binary.DefaultConfig())

	idx, err := ReadChunkIndexV2(r, 0, 2)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if idx == nil {
		t.Error("expected non-nil index")
	}
	if len(idx.Entries) != 0 {
		t.Errorf("expected 0 entries, got %d", len(idx.Entries))
	}
}

func TestBTreeV2Constants(t *testing.T) {
	// Verify the B-tree v2 type constants
	if BTreeV2TypeChunkNoFilter != 10 {
		t.Errorf("expected BTreeV2TypeChunkNoFilter=10, got %d", BTreeV2TypeChunkNoFilter)
	}
	if BTreeV2TypeChunkWithFilter != 11 {
		t.Errorf("expected BTreeV2TypeChunkWithFilter=11, got %d", BTreeV2TypeChunkWithFilter)
	}
}
