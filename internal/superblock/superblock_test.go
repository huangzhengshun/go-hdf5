package superblock

import (
	"bytes"
	"encoding/binary"
	"testing"

	binpkg "github.com/huangzhengshun/go-hdf5/internal/binary"
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

func TestSignature(t *testing.T) {
	expected := []byte{0x89, 'H', 'D', 'F', '\r', '\n', 0x1a, '\n'}
	if !bytes.Equal(Signature, expected) {
		t.Errorf("Signature mismatch: got %v, expected %v", Signature, expected)
	}
}

func TestReadNotHDF5(t *testing.T) {
	data := make(bytesReaderAt, 4096)
	// No HDF5 signature

	_, err := Read(data)
	if err != ErrNotHDF5 {
		t.Errorf("expected ErrNotHDF5, got %v", err)
	}
}

func TestReadUnsupportedVersion(t *testing.T) {
	// Create minimal data with signature but unsupported version
	data := make(bytesReaderAt, 256)
	copy(data[0:8], Signature)
	data[8] = 99 // Unsupported version

	_, err := Read(data)
	if err != ErrUnsupportedVersion {
		t.Errorf("expected ErrUnsupportedVersion, got %v", err)
	}
}

func TestReadV2SuperblockMinimal(t *testing.T) {
	// Construct a minimal valid v2 superblock
	var buf bytes.Buffer

	// Signature (8 bytes)
	buf.Write(Signature)

	// Version (1 byte)
	buf.WriteByte(2)

	// Size of offsets (1 byte)
	buf.WriteByte(8)

	// Size of lengths (1 byte)
	buf.WriteByte(8)

	// File consistency flags (1 byte)
	buf.WriteByte(0)

	// Base address (8 bytes)
	binary.Write(&buf, binary.LittleEndian, uint64(0))

	// Superblock extension address (8 bytes) - undefined
	binary.Write(&buf, binary.LittleEndian, uint64(0xFFFFFFFFFFFFFFFF))

	// EOF address (8 bytes)
	binary.Write(&buf, binary.LittleEndian, uint64(1024))

	// Root group object header address (8 bytes)
	binary.Write(&buf, binary.LittleEndian, uint64(96))

	// Calculate and append checksum
	data := buf.Bytes()
	checksum := binpkg.Lookup3Checksum(data)
	binary.Write(&buf, binary.LittleEndian, checksum)

	// Pad to reasonable size
	fullData := make(bytesReaderAt, 256)
	copy(fullData, buf.Bytes())

	sb, err := Read(fullData)
	if err != nil {
		t.Fatalf("Read failed: %v", err)
	}

	if sb.Version != 2 {
		t.Errorf("expected version 2, got %d", sb.Version)
	}
	if sb.OffsetSize != 8 {
		t.Errorf("expected offset size 8, got %d", sb.OffsetSize)
	}
	if sb.LengthSize != 8 {
		t.Errorf("expected length size 8, got %d", sb.LengthSize)
	}
	if sb.BaseAddress != 0 {
		t.Errorf("expected base address 0, got %d", sb.BaseAddress)
	}
	if sb.EOFAddress != 1024 {
		t.Errorf("expected EOF address 1024, got %d", sb.EOFAddress)
	}
	if sb.RootGroupAddress != 96 {
		t.Errorf("expected root group address 96, got %d", sb.RootGroupAddress)
	}
	if sb.FileOffset != 0 {
		t.Errorf("expected file offset 0, got %d", sb.FileOffset)
	}
}

func TestReadV2SuperblockWithOffset(t *testing.T) {
	// Test that superblock can be found at offset 512
	var buf bytes.Buffer

	// Signature
	buf.Write(Signature)
	buf.WriteByte(2) // Version
	buf.WriteByte(8) // Offset size
	buf.WriteByte(8) // Length size
	buf.WriteByte(0) // Flags

	binary.Write(&buf, binary.LittleEndian, uint64(0))    // Base
	binary.Write(&buf, binary.LittleEndian, uint64(0xFF)) // Ext (undefined)
	binary.Write(&buf, binary.LittleEndian, uint64(2048)) // EOF
	binary.Write(&buf, binary.LittleEndian, uint64(600))  // Root

	// Checksum
	data := buf.Bytes()
	checksum := binpkg.Lookup3Checksum(data)
	binary.Write(&buf, binary.LittleEndian, checksum)

	// Place at offset 512
	fullData := make(bytesReaderAt, 1024)
	copy(fullData[512:], buf.Bytes())

	sb, err := Read(fullData)
	if err != nil {
		t.Fatalf("Read failed: %v", err)
	}

	if sb.FileOffset != 512 {
		t.Errorf("expected file offset 512, got %d", sb.FileOffset)
	}
	if sb.RootGroupAddress != 600 {
		t.Errorf("expected root group address 600, got %d", sb.RootGroupAddress)
	}
}

func TestReadV2SuperblockChecksumFailure(t *testing.T) {
	var buf bytes.Buffer

	buf.Write(Signature)
	buf.WriteByte(2)
	buf.WriteByte(8)
	buf.WriteByte(8)
	buf.WriteByte(0)

	binary.Write(&buf, binary.LittleEndian, uint64(0))
	binary.Write(&buf, binary.LittleEndian, uint64(0xFF))
	binary.Write(&buf, binary.LittleEndian, uint64(1024))
	binary.Write(&buf, binary.LittleEndian, uint64(96))

	// Wrong checksum
	binary.Write(&buf, binary.LittleEndian, uint32(0xDEADBEEF))

	fullData := make(bytesReaderAt, 256)
	copy(fullData, buf.Bytes())

	_, err := Read(fullData)
	if err != ErrInvalidSuperblock {
		t.Errorf("expected ErrInvalidSuperblock for bad checksum, got %v", err)
	}
}

func TestReadV0SuperblockMinimal(t *testing.T) {
	// Construct a minimal valid v0 superblock
	fullData := make(bytesReaderAt, 256)

	// Signature (offset 0)
	copy(fullData[0:8], Signature)

	// Header bytes (offset 8)
	fullData[8] = 0   // Version
	fullData[9] = 0   // Free-space storage version
	fullData[10] = 0  // Root group symbol table entry version
	fullData[11] = 0  // Reserved
	fullData[12] = 0  // Shared header message format version
	fullData[13] = 8  // Size of offsets
	fullData[14] = 8  // Size of lengths
	fullData[15] = 0  // Reserved

	// Group leaf node K (offset 16) - little endian uint16
	fullData[16] = 4
	fullData[17] = 0

	// Group internal node K (offset 18) - little endian uint16
	fullData[18] = 16
	fullData[19] = 0

	// File consistency flags (offset 20) - 4 bytes
	fullData[20] = 0
	fullData[21] = 0
	fullData[22] = 0
	fullData[23] = 0

	// Addresses start at offset 24, each is 8 bytes
	// Base address (offset 24)
	// Leave as 0

	// Free-space info address (offset 32) - skip

	// EOF address (offset 40)
	binary.LittleEndian.PutUint64(fullData[40:48], 1024)

	// Driver info block address (offset 48) - skip

	// Root group symbol table entry (offset 56)
	// Link name offset (8 bytes) - skip
	// Object header address (offset 64)
	binary.LittleEndian.PutUint64(fullData[64:72], 128)

	sb, err := Read(fullData)
	if err != nil {
		t.Fatalf("Read failed: %v", err)
	}

	if sb.Version != 0 {
		t.Errorf("expected version 0, got %d", sb.Version)
	}
	if sb.OffsetSize != 8 {
		t.Errorf("expected offset size 8, got %d", sb.OffsetSize)
	}
	if sb.GroupLeafNodeK != 4 {
		t.Errorf("expected group leaf node K 4, got %d", sb.GroupLeafNodeK)
	}
	if sb.GroupInternalNodeK != 16 {
		t.Errorf("expected group internal node K 16, got %d", sb.GroupInternalNodeK)
	}
	if sb.EOFAddress != 1024 {
		t.Errorf("expected EOF address 1024, got %d", sb.EOFAddress)
	}
	if sb.RootGroupAddress != 128 {
		t.Errorf("expected root group address 128, got %d", sb.RootGroupAddress)
	}
}

func TestSuperblockReaderConfig(t *testing.T) {
	sb := &Superblock{
		Version:    2,
		OffsetSize: 8,
		LengthSize: 8,
		ByteOrder:  binary.LittleEndian,
	}

	cfg := sb.ReaderConfig()

	if cfg.OffsetSize != 8 {
		t.Errorf("expected offset size 8, got %d", cfg.OffsetSize)
	}
	if cfg.LengthSize != 8 {
		t.Errorf("expected length size 8, got %d", cfg.LengthSize)
	}
	if cfg.ByteOrder != binary.LittleEndian {
		t.Error("expected little-endian byte order")
	}
}

func TestBytesEqual(t *testing.T) {
	tests := []struct {
		a, b     []byte
		expected bool
	}{
		{[]byte{1, 2, 3}, []byte{1, 2, 3}, true},
		{[]byte{1, 2, 3}, []byte{1, 2, 4}, false},
		{[]byte{1, 2, 3}, []byte{1, 2}, false},
		{[]byte{}, []byte{}, true},
		{nil, nil, true},
		{nil, []byte{}, true},
	}

	for _, tt := range tests {
		result := bytesEqual(tt.a, tt.b)
		if result != tt.expected {
			t.Errorf("bytesEqual(%v, %v): expected %v, got %v",
				tt.a, tt.b, tt.expected, result)
		}
	}
}
