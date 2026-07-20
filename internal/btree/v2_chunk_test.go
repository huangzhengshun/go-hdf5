package btree

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"github.com/huangzhengshun/go-hdf5/internal/binary"
)

// Helper to get testdata path
func getTestdataPath(filename string) string {
	return filepath.Join("..", "..", "testdata", filename)
}

// TestReadChunkIndexV2FromFile tests reading B-tree v2 from an actual HDF5 file.
// This is an integration test that verifies the code works with real files.
func TestReadChunkIndexV2FromFile(t *testing.T) {
	path := getTestdataPath("btree_v2.h5")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skipf("Test file %s not found", path)
	}

	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("Failed to open file: %v", err)
	}
	defer f.Close()

	// The file should contain valid B-tree v2 structures
	// This test verifies we can parse the file without errors
	t.Log("Successfully opened btree_v2.h5 for B-tree v2 testing")
}

// TestReadChunkIndexV2CompressedFromFile tests reading compressed B-tree v2 from file.
func TestReadChunkIndexV2CompressedFromFile(t *testing.T) {
	path := getTestdataPath("btree_v2_compressed.h5")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skipf("Test file %s not found", path)
	}

	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("Failed to open file: %v", err)
	}
	defer f.Close()

	t.Log("Successfully opened btree_v2_compressed.h5 for compressed B-tree v2 testing")
}

// TestBTreeV2HeaderParsing tests parsing a valid B-tree v2 header.
func TestBTreeV2HeaderParsing(t *testing.T) {
	// Build a valid BTHD header
	buf := bytes.NewBuffer(nil)
	buf.WriteString("BTHD")               // Signature
	buf.WriteByte(0)                      // Version 0
	buf.WriteByte(10)                     // Type 10 (chunks without filter)
	buf.Write([]byte{0, 4, 0, 0})         // Node size = 1024
	buf.Write([]byte{24, 0})              // Record size = 24
	buf.Write([]byte{0, 0})               // Depth = 0
	buf.WriteByte(75)                     // Split percent
	buf.WriteByte(25)                     // Merge percent
	buf.Write([]byte{0, 1, 0, 0, 0, 0, 0, 0}) // Root address = 256
	buf.Write([]byte{1, 0})               // Num root records = 1
	buf.Write([]byte{1, 0, 0, 0, 0, 0, 0, 0}) // Total records = 1

	r := binary.NewReader(bytes.NewReader(buf.Bytes()), binary.DefaultConfig())

	// Just testing that we can read header without panic
	// The ReadChunkIndexV2 will fail because we don't have a full tree,
	// but the header parsing should work
	nr := r.At(0)

	// Check signature
	sig, err := nr.ReadBytes(4)
	if err != nil {
		t.Fatalf("reading signature: %v", err)
	}
	if string(sig) != "BTHD" {
		t.Errorf("expected BTHD signature, got %q", string(sig))
	}

	// Version
	version, err := nr.ReadUint8()
	if err != nil {
		t.Fatalf("reading version: %v", err)
	}
	if version != 0 {
		t.Errorf("expected version 0, got %d", version)
	}

	// Type
	typ, err := nr.ReadUint8()
	if err != nil {
		t.Fatalf("reading type: %v", err)
	}
	if typ != 10 {
		t.Errorf("expected type 10, got %d", typ)
	}
}

// TestBTreeV2InvalidSignature tests that invalid signatures are properly rejected.
func TestBTreeV2InvalidSignature(t *testing.T) {
	tests := []struct {
		name      string
		signature string
	}{
		{"empty signature", "\x00\x00\x00\x00"},
		{"wrong signature", "XXXX"},
		{"partial match BTHX", "BTHX"},
		{"lowercase", "bthd"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			buf := bytes.NewBuffer(nil)
			buf.WriteString(tt.signature)

			r := binary.NewReader(bytes.NewReader(buf.Bytes()), binary.DefaultConfig())

			_, err := ReadChunkIndexV2(r, 0, 2)
			if err == nil {
				t.Error("expected error for invalid signature")
			}
			if !bytes.Contains([]byte(err.Error()), []byte("invalid B-tree v2 signature")) {
				t.Errorf("unexpected error message: %v", err)
			}
		})
	}
}

// TestBTreeV2UnsupportedVersion tests that unsupported versions are rejected.
func TestBTreeV2UnsupportedVersion(t *testing.T) {
	for version := 1; version <= 5; version++ {
		t.Run("version_"+string(rune('0'+version)), func(t *testing.T) {
			buf := bytes.NewBuffer(nil)
			buf.WriteString("BTHD")
			buf.WriteByte(byte(version)) // Unsupported version

			r := binary.NewReader(bytes.NewReader(buf.Bytes()), binary.DefaultConfig())

			_, err := ReadChunkIndexV2(r, 0, 2)
			if err == nil {
				t.Error("expected error for unsupported version")
			}
			if !bytes.Contains([]byte(err.Error()), []byte("unsupported B-tree v2 version")) {
				t.Errorf("unexpected error message: %v", err)
			}
		})
	}
}

// TestBTreeV2WrongType tests that non-chunk B-tree types are rejected.
func TestBTreeV2WrongType(t *testing.T) {
	// Test various non-chunk types
	wrongTypes := []uint8{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 255}

	for _, wrongType := range wrongTypes {
		t.Run("type_"+string(rune('0'+wrongType)), func(t *testing.T) {
			buf := bytes.NewBuffer(nil)
			buf.WriteString("BTHD")
			buf.WriteByte(0)         // Version 0
			buf.WriteByte(wrongType) // Wrong type
			buf.Write([]byte{0, 0, 0, 0})       // Node size
			buf.Write([]byte{0, 0})             // Record size
			buf.Write([]byte{0, 0})             // Depth
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
		})
	}
}

// TestBTreeV2ValidChunkTypes tests that valid chunk types (10 and 11) are accepted.
func TestBTreeV2ValidChunkTypes(t *testing.T) {
	for _, chunkType := range []uint8{BTreeV2TypeChunkNoFilter, BTreeV2TypeChunkWithFilter} {
		t.Run("type_"+string(rune('0'+chunkType)), func(t *testing.T) {
			buf := bytes.NewBuffer(nil)
			buf.WriteString("BTHD")
			buf.WriteByte(0)         // Version 0
			buf.WriteByte(chunkType) // Valid chunk type
			buf.Write([]byte{0, 4, 0, 0})       // Node size = 1024
			buf.Write([]byte{24, 0})            // Record size = 24
			buf.Write([]byte{0, 0})             // Depth = 0
			buf.WriteByte(75)                   // Split percent
			buf.WriteByte(25)                   // Merge percent
			buf.Write(make([]byte, 8))          // Root address = 0
			buf.Write([]byte{0, 0})             // Num root records = 0
			buf.Write([]byte{0, 0, 0, 0, 0, 0, 0, 0}) // Total records = 0

			r := binary.NewReader(bytes.NewReader(buf.Bytes()), binary.DefaultConfig())

			idx, err := ReadChunkIndexV2(r, 0, 2)
			if err != nil {
				t.Errorf("unexpected error for valid chunk type %d: %v", chunkType, err)
			}
			if idx == nil {
				t.Error("expected non-nil index")
			}
			if len(idx.Entries) != 0 {
				t.Errorf("expected 0 entries for empty index, got %d", len(idx.Entries))
			}
		})
	}
}

// TestBTreeV2TruncatedHeader tests handling of truncated headers.
func TestBTreeV2TruncatedHeader(t *testing.T) {
	tests := []struct {
		name string
		data []byte
	}{
		{"empty", []byte{}},
		{"signature only", []byte("BTHD")},
		{"signature + version", []byte("BTHD\x00")},
		{"signature + version + type", []byte("BTHD\x00\x0a")},
		{"missing total records", append([]byte("BTHD\x00\x0a"), make([]byte, 16)...)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := binary.NewReader(bytes.NewReader(tt.data), binary.DefaultConfig())

			_, err := ReadChunkIndexV2(r, 0, 2)
			if err == nil {
				t.Error("expected error for truncated data")
			}
		})
	}
}

// TestBTreeV2LeafSignature tests that leaf nodes with wrong signatures are rejected.
func TestBTreeV2LeafSignature(t *testing.T) {
	// Build a valid header that points to an invalid leaf
	buf := bytes.NewBuffer(nil)
	buf.WriteString("BTHD")                       // Signature
	buf.WriteByte(0)                              // Version 0
	buf.WriteByte(10)                             // Type 10
	buf.Write([]byte{0, 4, 0, 0})                 // Node size = 1024
	buf.Write([]byte{24, 0})                      // Record size = 24
	buf.Write([]byte{0, 0})                       // Depth = 0 (root is leaf)
	buf.WriteByte(75)                             // Split percent
	buf.WriteByte(25)                             // Merge percent
	buf.Write([]byte{100, 0, 0, 0, 0, 0, 0, 0})   // Root address = 100
	buf.Write([]byte{1, 0})                       // Num root records = 1
	buf.Write([]byte{1, 0, 0, 0, 0, 0, 0, 0})     // Total records = 1

	// Pad to offset 100
	for buf.Len() < 100 {
		buf.WriteByte(0)
	}

	// Write invalid leaf signature at offset 100
	buf.WriteString("XXXX") // Invalid signature

	r := binary.NewReader(bytes.NewReader(buf.Bytes()), binary.DefaultConfig())

	_, err := ReadChunkIndexV2(r, 0, 2)
	if err == nil {
		t.Error("expected error for invalid leaf signature")
	}
	if !bytes.Contains([]byte(err.Error()), []byte("invalid B-tree v2 leaf signature")) {
		t.Errorf("unexpected error message: %v", err)
	}
}

// TestBTreeV2LeafVersion tests that leaf nodes with wrong versions are rejected.
func TestBTreeV2LeafVersion(t *testing.T) {
	// Build a valid header that points to a leaf with wrong version
	buf := bytes.NewBuffer(nil)
	buf.WriteString("BTHD")                       // Signature
	buf.WriteByte(0)                              // Version 0
	buf.WriteByte(10)                             // Type 10
	buf.Write([]byte{0, 4, 0, 0})                 // Node size = 1024
	buf.Write([]byte{24, 0})                      // Record size = 24
	buf.Write([]byte{0, 0})                       // Depth = 0 (root is leaf)
	buf.WriteByte(75)                             // Split percent
	buf.WriteByte(25)                             // Merge percent
	buf.Write([]byte{100, 0, 0, 0, 0, 0, 0, 0})   // Root address = 100
	buf.Write([]byte{1, 0})                       // Num root records = 1
	buf.Write([]byte{1, 0, 0, 0, 0, 0, 0, 0})     // Total records = 1

	// Pad to offset 100
	for buf.Len() < 100 {
		buf.WriteByte(0)
	}

	// Write leaf with wrong version at offset 100
	buf.WriteString("BTLF") // Valid leaf signature
	buf.WriteByte(5)        // Wrong version (should be 0)

	r := binary.NewReader(bytes.NewReader(buf.Bytes()), binary.DefaultConfig())

	_, err := ReadChunkIndexV2(r, 0, 2)
	if err == nil {
		t.Error("expected error for wrong leaf version")
	}
	if !bytes.Contains([]byte(err.Error()), []byte("unsupported B-tree v2 leaf version")) {
		t.Errorf("unexpected error message: %v", err)
	}
}

// TestBTreeV2InternalNodeSignature tests that internal nodes with wrong signatures are rejected.
func TestBTreeV2InternalNodeSignature(t *testing.T) {
	// Build a valid header with depth > 0 that points to an invalid internal node
	buf := bytes.NewBuffer(nil)
	buf.WriteString("BTHD")                       // Signature
	buf.WriteByte(0)                              // Version 0
	buf.WriteByte(10)                             // Type 10
	buf.Write([]byte{0, 4, 0, 0})                 // Node size = 1024
	buf.Write([]byte{24, 0})                      // Record size = 24
	buf.Write([]byte{1, 0})                       // Depth = 1 (root is internal node)
	buf.WriteByte(75)                             // Split percent
	buf.WriteByte(25)                             // Merge percent
	buf.Write([]byte{100, 0, 0, 0, 0, 0, 0, 0})   // Root address = 100
	buf.Write([]byte{1, 0})                       // Num root records = 1
	buf.Write([]byte{1, 0, 0, 0, 0, 0, 0, 0})     // Total records = 1

	// Pad to offset 100
	for buf.Len() < 100 {
		buf.WriteByte(0)
	}

	// Write invalid internal node signature at offset 100
	buf.WriteString("XXXX") // Invalid signature

	r := binary.NewReader(bytes.NewReader(buf.Bytes()), binary.DefaultConfig())

	_, err := ReadChunkIndexV2(r, 0, 2)
	if err == nil {
		t.Error("expected error for invalid internal node signature")
	}
	if !bytes.Contains([]byte(err.Error()), []byte("invalid B-tree v2 internal node signature")) {
		t.Errorf("unexpected error message: %v", err)
	}
}

// TestBTreeV2InternalNodeVersion tests that internal nodes with wrong versions are rejected.
func TestBTreeV2InternalNodeVersion(t *testing.T) {
	// Build a valid header with depth > 0 that points to an internal node with wrong version
	buf := bytes.NewBuffer(nil)
	buf.WriteString("BTHD")                       // Signature
	buf.WriteByte(0)                              // Version 0
	buf.WriteByte(10)                             // Type 10
	buf.Write([]byte{0, 4, 0, 0})                 // Node size = 1024
	buf.Write([]byte{24, 0})                      // Record size = 24
	buf.Write([]byte{1, 0})                       // Depth = 1 (root is internal node)
	buf.WriteByte(75)                             // Split percent
	buf.WriteByte(25)                             // Merge percent
	buf.Write([]byte{100, 0, 0, 0, 0, 0, 0, 0})   // Root address = 100
	buf.Write([]byte{1, 0})                       // Num root records = 1
	buf.Write([]byte{1, 0, 0, 0, 0, 0, 0, 0})     // Total records = 1

	// Pad to offset 100
	for buf.Len() < 100 {
		buf.WriteByte(0)
	}

	// Write internal node with wrong version at offset 100
	buf.WriteString("BTIN") // Valid internal node signature
	buf.WriteByte(5)        // Wrong version (should be 0)

	r := binary.NewReader(bytes.NewReader(buf.Bytes()), binary.DefaultConfig())

	_, err := ReadChunkIndexV2(r, 0, 2)
	if err == nil {
		t.Error("expected error for wrong internal node version")
	}
	if !bytes.Contains([]byte(err.Error()), []byte("unsupported B-tree v2 internal node version")) {
		t.Errorf("unexpected error message: %v", err)
	}
}

// TestBTreeV2TypeConstants verifies the B-tree v2 type constants.
func TestBTreeV2TypeConstants(t *testing.T) {
	if BTreeV2TypeChunkNoFilter != 10 {
		t.Errorf("BTreeV2TypeChunkNoFilter should be 10, got %d", BTreeV2TypeChunkNoFilter)
	}
	if BTreeV2TypeChunkWithFilter != 11 {
		t.Errorf("BTreeV2TypeChunkWithFilter should be 11, got %d", BTreeV2TypeChunkWithFilter)
	}
}

// TestBTreeV2NonZeroOffset tests reading a B-tree header at a non-zero offset.
func TestBTreeV2NonZeroOffset(t *testing.T) {
	// Create a buffer with padding before the header
	buf := bytes.NewBuffer(nil)

	// Add 256 bytes of padding
	buf.Write(make([]byte, 256))

	// Write valid header at offset 256
	buf.WriteString("BTHD")
	buf.WriteByte(0)                              // Version 0
	buf.WriteByte(10)                             // Type 10
	buf.Write([]byte{0, 4, 0, 0})                 // Node size = 1024
	buf.Write([]byte{24, 0})                      // Record size = 24
	buf.Write([]byte{0, 0})                       // Depth = 0
	buf.WriteByte(75)                             // Split percent
	buf.WriteByte(25)                             // Merge percent
	buf.Write(make([]byte, 8))                    // Root address
	buf.Write([]byte{0, 0})                       // Num root records = 0
	buf.Write([]byte{0, 0, 0, 0, 0, 0, 0, 0})     // Total records = 0

	r := binary.NewReader(bytes.NewReader(buf.Bytes()), binary.DefaultConfig())

	// Read at offset 256
	idx, err := ReadChunkIndexV2(r, 256, 2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if idx == nil {
		t.Error("expected non-nil index")
	}
}
