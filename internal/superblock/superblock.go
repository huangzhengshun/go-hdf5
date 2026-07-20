// Package superblock handles parsing of HDF5 superblock structures.
//
// The superblock is the entry point for any HDF5 file, containing critical
// metadata like file version, offset/length sizes, and the root group address.
package superblock

import (
	"encoding/binary"
	"errors"
	"io"

	binpkg "github.com/huangzhengshun/go-hdf5/internal/binary"
)

// HDF5 file signature: 0x89 H D F \r \n 0x1a \n
var Signature = []byte{0x89, 'H', 'D', 'F', '\r', '\n', 0x1a, '\n'}

// Possible superblock locations (searched in order)
var superblockOffsets = []int64{0, 512, 1024, 2048}

// Errors
var (
	ErrNotHDF5            = errors.New("not an HDF5 file: signature not found")
	ErrUnsupportedVersion = errors.New("unsupported superblock version")
	ErrInvalidSuperblock  = errors.New("invalid superblock structure")
)

// Superblock contains the essential HDF5 file metadata.
type Superblock struct {
	// Version is the superblock format version (0, 1, 2, or 3)
	Version uint8

	// OffsetSize is the number of bytes used for file offsets (2, 4, or 8)
	OffsetSize uint8

	// LengthSize is the number of bytes used for lengths (2, 4, or 8)
	LengthSize uint8

	// FileConsistencyFlags contains file consistency information (v2/v3 only)
	FileConsistencyFlags uint8

	// BaseAddress is the absolute file address of byte 0 of the file
	// (usually 0, but can be non-zero for embedded HDF5 files)
	BaseAddress uint64

	// SuperblockExtensionAddress is the address of the superblock extension
	// (v2/v3 only, undefined if not present)
	SuperblockExtensionAddress uint64

	// EOFAddress is the end-of-file address (logical EOF)
	EOFAddress uint64

	// RootGroupAddress is the address of the root group object header
	RootGroupAddress uint64

	// V0/V1 specific fields
	GroupLeafNodeK        uint16 // 1/2 rank of B-tree leaf nodes for group nodes
	GroupInternalNodeK    uint16 // 1/2 rank of B-tree internal nodes for group nodes
	IndexedStorageK       uint16 // 1/2 rank of B-tree nodes for indexed storage (v1 only)
	FreeSpaceManagerVersion uint8  // (v0/v1 only)
	RootGroupSymbolTableAddress uint64 // Address of root group symbol table entry (v0/v1)
	RootGroupBTreeAddress       uint64 // B-tree address from root group scratch pad (v0/v1)
	RootGroupLocalHeapAddress   uint64 // Local heap address from root group scratch pad (v0/v1)

	// Computed/derived fields
	ByteOrder binary.ByteOrder // Always little-endian for HDF5

	// Location where superblock was found
	FileOffset int64
}

// Read locates and parses the superblock from an HDF5 file.
// It searches for the HDF5 signature at standard offsets and parses
// the appropriate superblock version.
func Read(r io.ReaderAt) (*Superblock, error) {
	sigBuf := make([]byte, 8)

	for _, offset := range superblockOffsets {
		if _, err := r.ReadAt(sigBuf, offset); err != nil {
			if err == io.EOF {
				continue
			}
			return nil, err
		}

		if !bytesEqual(sigBuf, Signature) {
			continue
		}

		// Found signature, read version byte
		verBuf := make([]byte, 1)
		if _, err := r.ReadAt(verBuf, offset+8); err != nil {
			return nil, err
		}
		version := verBuf[0]

		var sb *Superblock
		var err error

		switch version {
		case 0:
			sb, err = readV0(r, offset)
		case 1:
			sb, err = readV1(r, offset)
		case 2:
			sb, err = readV2(r, offset)
		case 3:
			sb, err = readV3(r, offset)
		default:
			return nil, ErrUnsupportedVersion
		}

		if err != nil {
			return nil, err
		}

		sb.FileOffset = offset
		sb.ByteOrder = binary.LittleEndian // HDF5 is always little-endian
		return sb, nil
	}

	return nil, ErrNotHDF5
}

// ReaderConfig returns a binary.Config for creating readers based on this superblock.
func (sb *Superblock) ReaderConfig() binpkg.Config {
	return binpkg.Config{
		ByteOrder:  sb.ByteOrder,
		OffsetSize: int(sb.OffsetSize),
		LengthSize: int(sb.LengthSize),
	}
}

// bytesEqual compares two byte slices for equality.
func bytesEqual(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// decodeUint decodes a variable-width unsigned integer in little-endian order.
func decodeUint(buf []byte, size int) uint64 {
	switch size {
	case 2:
		return uint64(binary.LittleEndian.Uint16(buf))
	case 4:
		return uint64(binary.LittleEndian.Uint32(buf))
	case 8:
		return binary.LittleEndian.Uint64(buf)
	default:
		// Handle arbitrary sizes
		var val uint64
		for i := size - 1; i >= 0; i-- {
			val = (val << 8) | uint64(buf[i])
		}
		return val
	}
}
