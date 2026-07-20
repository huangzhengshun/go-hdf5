package superblock

import (
	"io"
)

/*
Version 0 Superblock Layout:
Offset  Size  Description
0       8     Signature
8       1     Version (0)
9       1     Free-space storage version
10      1     Root group symbol table entry version
11      1     Reserved
12      1     Shared header message format version
13      1     Size of offsets
14      1     Size of lengths
15      1     Reserved
16      2     Group leaf node K
18      2     Group internal node K
20      4     File consistency flags
24      O     Base address
24+O    O     Free-space info address
24+2O   O     EOF address
24+3O   O     Driver info block address
24+4O   var   Root group symbol table entry

Where O = size of offsets

Root Group Symbol Table Entry:
0       O     Link name offset (into local heap, always 0)
O       O     Object header address
2O      32    Scratch-pad space (cache info)
*/

// readV0 parses a version 0 superblock.
func readV0(r io.ReaderAt, offset int64) (*Superblock, error) {
	// Read fixed-size header portion (first 24 bytes after signature)
	header := make([]byte, 16)
	if _, err := r.ReadAt(header, offset+8); err != nil {
		return nil, err
	}

	sb := &Superblock{
		Version:               header[0], // 0
		FreeSpaceManagerVersion: header[1],
		// header[2] = root group symbol table entry version (must be 0)
		// header[3] = reserved
		// header[4] = shared header message format version
		OffsetSize:           header[5],
		LengthSize:           header[6],
		// header[7] = reserved
		GroupLeafNodeK:       uint16(header[8]) | uint16(header[9])<<8,
		GroupInternalNodeK:   uint16(header[10]) | uint16(header[11])<<8,
		// header[12:16] = file consistency flags (4 bytes)
	}

	osize := int(sb.OffsetSize)

	// Position after fixed header
	pos := offset + 24
	addrBuf := make([]byte, osize)

	// Base address
	if _, err := r.ReadAt(addrBuf, pos); err != nil {
		return nil, err
	}
	sb.BaseAddress = decodeUint(addrBuf, osize)
	pos += int64(osize)

	// Free-space info address (skip)
	pos += int64(osize)

	// EOF address
	if _, err := r.ReadAt(addrBuf, pos); err != nil {
		return nil, err
	}
	sb.EOFAddress = decodeUint(addrBuf, osize)
	pos += int64(osize)

	// Driver info block address (skip)
	pos += int64(osize)

	// Root group symbol table entry
	// Skip link name offset (first O bytes)
	pos += int64(osize)

	// Read object header address
	if _, err := r.ReadAt(addrBuf, pos); err != nil {
		return nil, err
	}
	sb.RootGroupAddress = decodeUint(addrBuf, osize)
	sb.RootGroupSymbolTableAddress = sb.RootGroupAddress
	pos += int64(osize)

	// Read cache type (4 bytes)
	cacheTypeBuf := make([]byte, 4)
	if _, err := r.ReadAt(cacheTypeBuf, pos); err != nil {
		return nil, err
	}
	cacheType := decodeUint(cacheTypeBuf, 4)
	pos += 8 // Skip cache type (4) + reserved (4)

	// If cache type is 1 (group), read B-tree and local heap addresses from scratch pad
	if cacheType == 1 {
		// B-tree address
		if _, err := r.ReadAt(addrBuf, pos); err != nil {
			return nil, err
		}
		sb.RootGroupBTreeAddress = decodeUint(addrBuf, osize)
		pos += int64(osize)

		// Local heap address
		if _, err := r.ReadAt(addrBuf, pos); err != nil {
			return nil, err
		}
		sb.RootGroupLocalHeapAddress = decodeUint(addrBuf, osize)
	}

	return sb, nil
}

// readV1 parses a version 1 superblock.
// Version 1 is similar to version 0 but includes indexed storage K value.
func readV1(r io.ReaderAt, offset int64) (*Superblock, error) {
	// Read fixed-size header portion
	header := make([]byte, 16)
	if _, err := r.ReadAt(header, offset+8); err != nil {
		return nil, err
	}

	sb := &Superblock{
		Version:               header[0], // 1
		FreeSpaceManagerVersion: header[1],
		OffsetSize:           header[5],
		LengthSize:           header[6],
		GroupLeafNodeK:       uint16(header[8]) | uint16(header[9])<<8,
		GroupInternalNodeK:   uint16(header[10]) | uint16(header[11])<<8,
	}

	osize := int(sb.OffsetSize)

	// Read indexed storage K (2 bytes at offset 24)
	kBuf := make([]byte, 2)
	if _, err := r.ReadAt(kBuf, offset+24); err != nil {
		return nil, err
	}
	sb.IndexedStorageK = uint16(kBuf[0]) | uint16(kBuf[1])<<8

	// Position after indexed storage K + 2 reserved bytes
	pos := offset + 28
	addrBuf := make([]byte, osize)

	// Base address
	if _, err := r.ReadAt(addrBuf, pos); err != nil {
		return nil, err
	}
	sb.BaseAddress = decodeUint(addrBuf, osize)
	pos += int64(osize)

	// Free-space info address (skip)
	pos += int64(osize)

	// EOF address
	if _, err := r.ReadAt(addrBuf, pos); err != nil {
		return nil, err
	}
	sb.EOFAddress = decodeUint(addrBuf, osize)
	pos += int64(osize)

	// Driver info block address (skip)
	pos += int64(osize)

	// Root group symbol table entry
	// Skip link name offset
	pos += int64(osize)

	// Read object header address
	if _, err := r.ReadAt(addrBuf, pos); err != nil {
		return nil, err
	}
	sb.RootGroupAddress = decodeUint(addrBuf, osize)
	sb.RootGroupSymbolTableAddress = sb.RootGroupAddress
	pos += int64(osize)

	// Read cache type (4 bytes)
	cacheTypeBuf := make([]byte, 4)
	if _, err := r.ReadAt(cacheTypeBuf, pos); err != nil {
		return nil, err
	}
	cacheType := decodeUint(cacheTypeBuf, 4)
	pos += 8 // Skip cache type (4) + reserved (4)

	// If cache type is 1 (group), read B-tree and local heap addresses from scratch pad
	if cacheType == 1 {
		// B-tree address
		if _, err := r.ReadAt(addrBuf, pos); err != nil {
			return nil, err
		}
		sb.RootGroupBTreeAddress = decodeUint(addrBuf, osize)
		pos += int64(osize)

		// Local heap address
		if _, err := r.ReadAt(addrBuf, pos); err != nil {
			return nil, err
		}
		sb.RootGroupLocalHeapAddress = decodeUint(addrBuf, osize)
	}

	return sb, nil
}
