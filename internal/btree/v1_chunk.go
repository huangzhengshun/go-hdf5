package btree

import (
	"fmt"

	"github.com/huangzhengshun/go-hdf5/internal/binary"
)

// ChunkEntry represents a chunk in the B-tree index.
type ChunkEntry struct {
	// Offset contains the chunk coordinates in dataset element space.
	// For a 2D dataset with chunks [10,10], chunk at offset [20,30]
	// covers elements [20:30, 30:40].
	Offset []uint64

	// FilterMask indicates which filters were disabled for this chunk.
	// Bit i = 1 means filter i was skipped.
	FilterMask uint32

	// Size is the size of the chunk data on disk (possibly compressed).
	Size uint32

	// Address is the file offset where chunk data is stored.
	Address uint64
}

// ChunkIndex contains all chunks for a dataset.
type ChunkIndex struct {
	// NDims is the number of dimensions (including the extra +1 for chunked storage).
	NDims int

	// Entries contains all chunk entries.
	Entries []ChunkEntry
}

// ReadChunkIndex reads a v1 B-tree chunk index.
// ndims is the number of dataset dimensions (not including the +1 used in B-tree keys).
func ReadChunkIndex(r *binary.Reader, btreeAddr uint64, ndims int) (*ChunkIndex, error) {
	index := &ChunkIndex{
		NDims: ndims,
	}

	entries, err := readChunkBTreeNode(r, btreeAddr, ndims)
	if err != nil {
		return nil, err
	}
	index.Entries = entries

	return index, nil
}

func readChunkBTreeNode(r *binary.Reader, address uint64, ndims int) ([]ChunkEntry, error) {
	nr := r.At(int64(address))

	// Check signature
	sig, err := nr.ReadBytes(4)
	if err != nil {
		return nil, fmt.Errorf("reading btree signature: %w", err)
	}
	if string(sig) != "TREE" {
		return nil, fmt.Errorf("invalid B-tree signature: got %q, expected \"TREE\"", string(sig))
	}

	// Node type (1 byte): 0 = group, 1 = chunk
	nodeType, err := nr.ReadUint8()
	if err != nil {
		return nil, err
	}
	if nodeType != 1 {
		return nil, fmt.Errorf("unexpected B-tree node type: %d (expected 1 for chunk)", nodeType)
	}

	// Node level (1 byte): 0 = leaf
	nodeLevel, err := nr.ReadUint8()
	if err != nil {
		return nil, err
	}

	// Entries used (2 bytes)
	entriesUsed, err := nr.ReadUint16()
	if err != nil {
		return nil, err
	}

	// Left sibling address
	_, err = nr.ReadOffset()
	if err != nil {
		return nil, err
	}

	// Right sibling address
	_, err = nr.ReadOffset()
	if err != nil {
		return nil, err
	}

	var entries []ChunkEntry

	if nodeLevel == 0 {
		// Leaf node - contains actual chunk entries
		// Key layout for chunked data (per HDF5 spec):
		// - Chunk size in bytes (4 bytes)
		// - Filter mask (4 bytes)
		// - Chunk offsets (ndims+1 values, each 8 bytes)
		// Child pointer:
		// - Address of chunk data (offset-sized)

		for i := uint16(0); i <= entriesUsed; i++ {
			// Read key
			chunkSize, err := nr.ReadUint32()
			if err != nil {
				return nil, fmt.Errorf("reading chunk size: %w", err)
			}

			filterMask, err := nr.ReadUint32()
			if err != nil {
				return nil, fmt.Errorf("reading filter mask: %w", err)
			}

			// Chunk offsets - HDF5 uses ndims+1 dimensions in the B-tree
			// The last dimension is typically the element size
			offsets := make([]uint64, ndims+1)
			for j := 0; j <= ndims; j++ {
				offsets[j], err = nr.ReadUint64()
				if err != nil {
					return nil, fmt.Errorf("reading chunk offset %d: %w", j, err)
				}
			}

			// For the last entry (i == entriesUsed), we only read the key
			// to know the upper bound, but there's no child pointer
			if i == entriesUsed {
				break
			}

			// Read child pointer (chunk data address)
			chunkAddr, err := nr.ReadOffset()
			if err != nil {
				return nil, fmt.Errorf("reading chunk address: %w", err)
			}

			// Only include chunks that have valid addresses
			if chunkAddr != 0xFFFFFFFFFFFFFFFF && chunkSize > 0 {
				entry := ChunkEntry{
					Offset:     offsets[:ndims], // Exclude the last dimension (element size)
					FilterMask: filterMask,
					Size:       chunkSize,
					Address:    chunkAddr,
				}
				entries = append(entries, entry)
			}
		}
	} else {
		// Internal node - recurse into children
		for i := uint16(0); i <= entriesUsed; i++ {
			// Read key (same format as leaf)
			_, err := nr.ReadUint32() // chunk size
			if err != nil {
				return nil, err
			}
			_, err = nr.ReadUint32() // filter mask
			if err != nil {
				return nil, err
			}
			for j := 0; j <= ndims; j++ {
				_, err = nr.ReadUint64() // offset
				if err != nil {
					return nil, err
				}
			}

			// For the last entry, no child pointer
			if i == entriesUsed {
				break
			}

			// Child pointer - address of child B-tree node
			childAddr, err := nr.ReadOffset()
			if err != nil {
				return nil, err
			}

			childEntries, err := readChunkBTreeNode(r, childAddr, ndims)
			if err != nil {
				return nil, err
			}
			entries = append(entries, childEntries...)
		}
	}

	return entries, nil
}

// FindChunk finds the chunk entry that contains the given offset.
// Returns nil if no chunk contains the offset.
func (idx *ChunkIndex) FindChunk(offset []uint64, chunkDims []uint32) *ChunkEntry {
	for i := range idx.Entries {
		entry := &idx.Entries[i]
		match := true
		for d := 0; d < len(offset) && d < len(entry.Offset); d++ {
			chunkStart := entry.Offset[d]
			chunkEnd := chunkStart + uint64(chunkDims[d])
			if offset[d] < chunkStart || offset[d] >= chunkEnd {
				match = false
				break
			}
		}
		if match {
			return entry
		}
	}
	return nil
}
