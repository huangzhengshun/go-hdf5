package superblock

import (
	"io"

	binpkg "github.com/huangzhengshun/go-hdf5/internal/binary"
)

/*
Version 2/3 Superblock Layout:
Offset  Size  Description
0       8     Signature
8       1     Version (2 or 3)
9       1     Size of offsets
10      1     Size of lengths
11      1     File consistency flags
12      O     Base address
12+O    O     Superblock extension address
12+2O   O     EOF address
12+3O   O     Root group object header address
12+4O   4     Superblock checksum (lookup3)

Where O = size of offsets

Version 2 and 3 are identical in structure. Version 3 adds support for
additional file consistency flags.
*/

// readV2 parses a version 2 superblock.
func readV2(r io.ReaderAt, offset int64) (*Superblock, error) {
	return readV2V3(r, offset, 2)
}

// readV3 parses a version 3 superblock.
func readV3(r io.ReaderAt, offset int64) (*Superblock, error) {
	return readV2V3(r, offset, 3)
}

// readV2V3 parses version 2 or 3 superblocks (same structure).
func readV2V3(r io.ReaderAt, offset int64, version uint8) (*Superblock, error) {
	// Read fixed header (4 bytes after signature)
	header := make([]byte, 4)
	if _, err := r.ReadAt(header, offset+8); err != nil {
		return nil, err
	}

	sb := &Superblock{
		Version:              header[0],
		OffsetSize:           header[1],
		LengthSize:           header[2],
		FileConsistencyFlags: header[3],
	}

	osize := int(sb.OffsetSize)
	pos := offset + 12
	addrBuf := make([]byte, osize)

	// Base address
	if _, err := r.ReadAt(addrBuf, pos); err != nil {
		return nil, err
	}
	sb.BaseAddress = decodeUint(addrBuf, osize)
	pos += int64(osize)

	// Superblock extension address
	if _, err := r.ReadAt(addrBuf, pos); err != nil {
		return nil, err
	}
	sb.SuperblockExtensionAddress = decodeUint(addrBuf, osize)
	pos += int64(osize)

	// EOF address
	if _, err := r.ReadAt(addrBuf, pos); err != nil {
		return nil, err
	}
	sb.EOFAddress = decodeUint(addrBuf, osize)
	pos += int64(osize)

	// Root group object header address
	if _, err := r.ReadAt(addrBuf, pos); err != nil {
		return nil, err
	}
	sb.RootGroupAddress = decodeUint(addrBuf, osize)
	pos += int64(osize)

	// Verify checksum (4 bytes)
	checksumStart := offset
	checksumEnd := pos
	checksumLen := int(checksumEnd - checksumStart)

	checksumData := make([]byte, checksumLen)
	if _, err := r.ReadAt(checksumData, checksumStart); err != nil {
		return nil, err
	}

	checksumBuf := make([]byte, 4)
	if _, err := r.ReadAt(checksumBuf, pos); err != nil {
		return nil, err
	}
	storedChecksum := uint32(checksumBuf[0]) | uint32(checksumBuf[1])<<8 |
		uint32(checksumBuf[2])<<16 | uint32(checksumBuf[3])<<24

	computedChecksum := binpkg.Lookup3Checksum(checksumData)
	if storedChecksum != computedChecksum {
		return nil, ErrInvalidSuperblock
	}

	return sb, nil
}
