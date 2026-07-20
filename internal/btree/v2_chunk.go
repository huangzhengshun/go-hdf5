package btree

import (
	"fmt"

	"github.com/huangzhengshun/go-hdf5/internal/binary"
)

// B-tree v2 types for chunked storage
const (
	// BTreeV2TypeChunkNoFilter is type 10: Chunk records without filter info
	BTreeV2TypeChunkNoFilter uint8 = 10
	// BTreeV2TypeChunkWithFilter is type 11: Chunk records with filter info
	BTreeV2TypeChunkWithFilter uint8 = 11
)

// btreeV2Header represents a B-tree v2 header (BTHD).
type btreeV2Header struct {
	Version        uint8
	Type           uint8
	NodeSize       uint32
	RecordSize     uint16
	Depth          uint16
	SplitPercent   uint8
	MergePercent   uint8
	RootAddr       uint64
	NumRootRecords uint16
	TotalRecords   uint64
}

// ReadChunkIndexV2 reads a v2 B-tree chunk index.
// ndims is the number of dataset dimensions.
func ReadChunkIndexV2(r *binary.Reader, btreeAddr uint64, ndims int) (*ChunkIndex, error) {
	// Read B-tree header
	header, err := readBTreeV2Header(r, btreeAddr)
	if err != nil {
		return nil, fmt.Errorf("reading B-tree v2 header: %w", err)
	}

	// Validate type is for chunked storage
	if header.Type != BTreeV2TypeChunkNoFilter && header.Type != BTreeV2TypeChunkWithFilter {
		return nil, fmt.Errorf("unexpected B-tree v2 type: %d (expected 10 or 11 for chunks)", header.Type)
	}

	index := &ChunkIndex{
		NDims: ndims,
	}

	// If depth is 0, root contains records directly (leaf)
	// If depth > 0, root is internal node pointing to children
	if header.TotalRecords == 0 {
		return index, nil // Empty index
	}

	hasFilter := header.Type == BTreeV2TypeChunkWithFilter

	var entries []ChunkEntry
	if header.Depth == 0 {
		// Root is a leaf node
		entries, err = readBTreeV2LeafRecords(r, header.RootAddr, int(header.NumRootRecords),
			header.RecordSize, ndims, hasFilter, r.OffsetSize())
	} else {
		// Root is internal node
		entries, err = readBTreeV2InternalNode(r, header.RootAddr, int(header.NumRootRecords),
			header, ndims, int(header.Depth), hasFilter)
	}

	if err != nil {
		return nil, err
	}
	index.Entries = entries

	return index, nil
}

// readBTreeV2Header reads the BTHD header.
func readBTreeV2Header(r *binary.Reader, address uint64) (*btreeV2Header, error) {
	nr := r.At(int64(address))

	// Check signature
	sig, err := nr.ReadBytes(4)
	if err != nil {
		return nil, fmt.Errorf("reading signature: %w", err)
	}
	if string(sig) != "BTHD" {
		return nil, fmt.Errorf("invalid B-tree v2 signature: %q (expected BTHD)", string(sig))
	}

	header := &btreeV2Header{}

	// Version (1 byte)
	header.Version, err = nr.ReadUint8()
	if err != nil {
		return nil, err
	}
	if header.Version != 0 {
		return nil, fmt.Errorf("unsupported B-tree v2 version: %d", header.Version)
	}

	// Type (1 byte)
	header.Type, err = nr.ReadUint8()
	if err != nil {
		return nil, err
	}

	// Node size (4 bytes)
	header.NodeSize, err = nr.ReadUint32()
	if err != nil {
		return nil, err
	}

	// Record size (2 bytes)
	header.RecordSize, err = nr.ReadUint16()
	if err != nil {
		return nil, err
	}

	// Depth (2 bytes)
	header.Depth, err = nr.ReadUint16()
	if err != nil {
		return nil, err
	}

	// Split percent (1 byte)
	header.SplitPercent, err = nr.ReadUint8()
	if err != nil {
		return nil, err
	}

	// Merge percent (1 byte)
	header.MergePercent, err = nr.ReadUint8()
	if err != nil {
		return nil, err
	}

	// Root node address (offset-sized)
	header.RootAddr, err = nr.ReadOffset()
	if err != nil {
		return nil, err
	}

	// Number of records in root (2 bytes)
	header.NumRootRecords, err = nr.ReadUint16()
	if err != nil {
		return nil, err
	}

	// Total number of records (length-sized)
	header.TotalRecords, err = nr.ReadLength()
	if err != nil {
		return nil, err
	}

	// Skip checksum (4 bytes) - we could verify it but not required for reading

	return header, nil
}

// readBTreeV2LeafRecords reads chunk records from a leaf node.
func readBTreeV2LeafRecords(r *binary.Reader, address uint64, numRecords int,
	recordSize uint16, ndims int, hasFilter bool, offsetSize int) ([]ChunkEntry, error) {

	nr := r.At(int64(address))

	// Check leaf signature
	sig, err := nr.ReadBytes(4)
	if err != nil {
		return nil, fmt.Errorf("reading leaf signature: %w", err)
	}
	if string(sig) != "BTLF" {
		return nil, fmt.Errorf("invalid B-tree v2 leaf signature: %q (expected BTLF)", string(sig))
	}

	// Version (1 byte)
	version, err := nr.ReadUint8()
	if err != nil {
		return nil, err
	}
	if version != 0 {
		return nil, fmt.Errorf("unsupported B-tree v2 leaf version: %d", version)
	}

	// Type (1 byte) - should match header type
	_, err = nr.ReadUint8()
	if err != nil {
		return nil, err
	}

	// Read records
	entries := make([]ChunkEntry, 0, numRecords)
	for i := 0; i < numRecords; i++ {
		entry, err := readChunkRecord(nr, ndims, hasFilter, offsetSize)
		if err != nil {
			return nil, fmt.Errorf("reading record %d: %w", i, err)
		}
		if entry.Address != 0 && entry.Address != 0xFFFFFFFFFFFFFFFF {
			entries = append(entries, entry)
		}
	}

	return entries, nil
}

// readBTreeV2InternalNode reads records from an internal node and recurses into children.
func readBTreeV2InternalNode(r *binary.Reader, address uint64, numRecords int,
	header *btreeV2Header, ndims int, depth int, hasFilter bool) ([]ChunkEntry, error) {

	nr := r.At(int64(address))

	// Check internal node signature
	sig, err := nr.ReadBytes(4)
	if err != nil {
		return nil, fmt.Errorf("reading internal node signature: %w", err)
	}
	if string(sig) != "BTIN" {
		return nil, fmt.Errorf("invalid B-tree v2 internal node signature: %q (expected BTIN)", string(sig))
	}

	// Version (1 byte)
	version, err := nr.ReadUint8()
	if err != nil {
		return nil, err
	}
	if version != 0 {
		return nil, fmt.Errorf("unsupported B-tree v2 internal node version: %d", version)
	}

	// Type (1 byte)
	_, err = nr.ReadUint8()
	if err != nil {
		return nil, err
	}

	var entries []ChunkEntry
	offsetSize := r.OffsetSize()

	// Internal nodes have: records interleaved with child pointers
	// Format: record[0], child[0], record[1], child[1], ..., record[n-1], child[n-1], child[n]
	// We read numRecords records and numRecords+1 child pointers

	for i := 0; i < numRecords; i++ {
		// Skip record (we don't need the keys, just child pointers)
		nr.Skip(int64(header.RecordSize))

		// Read child pointer
		childAddr, err := nr.ReadOffset()
		if err != nil {
			return nil, fmt.Errorf("reading child pointer %d: %w", i, err)
		}

		// Read number of records in child (2 bytes)
		childNumRecords, err := nr.ReadUint16()
		if err != nil {
			return nil, fmt.Errorf("reading child record count %d: %w", i, err)
		}

		// Recurse into child
		var childEntries []ChunkEntry
		if depth == 1 {
			// Child is a leaf
			childEntries, err = readBTreeV2LeafRecords(r, childAddr, int(childNumRecords),
				header.RecordSize, ndims, hasFilter, offsetSize)
		} else {
			// Child is another internal node
			childEntries, err = readBTreeV2InternalNode(r, childAddr, int(childNumRecords),
				header, ndims, depth-1, hasFilter)
		}
		if err != nil {
			return nil, fmt.Errorf("reading child node %d: %w", i, err)
		}
		entries = append(entries, childEntries...)
	}

	// Read the last child pointer (after all records)
	childAddr, err := nr.ReadOffset()
	if err != nil {
		return nil, fmt.Errorf("reading last child pointer: %w", err)
	}
	childNumRecords, err := nr.ReadUint16()
	if err != nil {
		return nil, fmt.Errorf("reading last child record count: %w", err)
	}

	// Recurse into last child
	var childEntries []ChunkEntry
	if depth == 1 {
		childEntries, err = readBTreeV2LeafRecords(r, childAddr, int(childNumRecords),
			header.RecordSize, ndims, hasFilter, offsetSize)
	} else {
		childEntries, err = readBTreeV2InternalNode(r, childAddr, int(childNumRecords),
			header, ndims, depth-1, hasFilter)
	}
	if err != nil {
		return nil, fmt.Errorf("reading last child node: %w", err)
	}
	entries = append(entries, childEntries...)

	return entries, nil
}

// readChunkRecord reads a single chunk record.
// For type 10 (no filter): scaled offsets + address
// For type 11 (with filter): address + chunk size + filter mask + scaled offsets
func readChunkRecord(nr *binary.Reader, ndims int, hasFilter bool, offsetSize int) (ChunkEntry, error) {
	var entry ChunkEntry
	var err error

	if hasFilter {
		// Type 11: With filter info
		// Address first
		entry.Address, err = nr.ReadOffset()
		if err != nil {
			return entry, err
		}

		// Chunk size (variable-length encoded as 8-bit field showing # of bytes, then value)
		// For simplicity, read as 4-byte uint (most common case)
		// Actually the spec says it's encoded size - let me read it properly
		chunkSizeLen, err := nr.ReadUint8()
		if err != nil {
			return entry, err
		}
		if chunkSizeLen > 0 {
			sizeBytes, err := nr.ReadBytes(int(chunkSizeLen))
			if err != nil {
				return entry, err
			}
			// Decode little-endian
			var size uint64
			for i := 0; i < len(sizeBytes); i++ {
				size |= uint64(sizeBytes[i]) << (8 * i)
			}
			entry.Size = uint32(size)
		}

		// Filter mask (4 bytes)
		entry.FilterMask, err = nr.ReadUint32()
		if err != nil {
			return entry, err
		}

		// Scaled offsets
		entry.Offset = make([]uint64, ndims)
		for d := 0; d < ndims; d++ {
			entry.Offset[d], err = nr.ReadUint64()
			if err != nil {
				return entry, err
			}
		}
	} else {
		// Type 10: No filter info
		// Scaled offsets first
		entry.Offset = make([]uint64, ndims)
		for d := 0; d < ndims; d++ {
			entry.Offset[d], err = nr.ReadUint64()
			if err != nil {
				return entry, err
			}
		}

		// Address
		entry.Address, err = nr.ReadOffset()
		if err != nil {
			return entry, err
		}

		// Size is not stored - will need to be calculated from chunk dimensions
		// Set to 0 to indicate it needs to be calculated
		entry.Size = 0
		entry.FilterMask = 0
	}

	return entry, nil
}
