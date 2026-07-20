// Package btree implements HDF5 B-tree structures.
package btree

import (
	"fmt"

	"github.com/huangzhengshun/go-hdf5/internal/binary"
	"github.com/huangzhengshun/go-hdf5/internal/heap"
)

// GroupEntry represents an entry in a v1 group B-tree.
type GroupEntry struct {
	Name          string
	ObjectAddress uint64
	LinkType      uint32 // 0=hard link, 1=soft link, 2=external (future)
	SoftLinkValue string // Target path for soft links
}

// Signature for v1 B-tree: "TREE"
var btreeSignature = []byte{'T', 'R', 'E', 'E'}

// Symbol table node signature: "SNOD"
var snodSignature = []byte{'S', 'N', 'O', 'D'}

// ReadGroupEntries reads all entries from a v1 group B-tree.
func ReadGroupEntries(r *binary.Reader, btreeAddr uint64, localHeap *heap.LocalHeap) ([]GroupEntry, error) {
	var entries []GroupEntry

	// Read B-tree node
	nodeEntries, err := readBTreeNode(r, btreeAddr, localHeap)
	if err != nil {
		return nil, err
	}
	entries = append(entries, nodeEntries...)

	return entries, nil
}

func readBTreeNode(r *binary.Reader, address uint64, localHeap *heap.LocalHeap) ([]GroupEntry, error) {
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
	if nodeType != 0 {
		return nil, fmt.Errorf("unexpected B-tree node type: %d (expected 0 for group)", nodeType)
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

	var entries []GroupEntry

	if nodeLevel == 0 {
		// Leaf node - contains keys pointing to symbol table nodes
		for i := uint16(0); i < entriesUsed; i++ {
			// Key for group node is just reserved (length-sized)
			_, err := nr.ReadLength()
			if err != nil {
				return nil, err
			}

			// Child pointer - address of symbol table node (SNOD)
			snodAddr, err := nr.ReadOffset()
			if err != nil {
				return nil, err
			}

			// Read symbol table node entries
			snodEntries, err := readSymbolTableNode(r, snodAddr, localHeap)
			if err != nil {
				return nil, fmt.Errorf("reading symbol table node: %w", err)
			}
			entries = append(entries, snodEntries...)
		}
	} else {
		// Internal node - recurse into children
		for i := uint16(0); i < entriesUsed; i++ {
			// Key
			_, err := nr.ReadLength()
			if err != nil {
				return nil, err
			}

			// Child pointer - address of child B-tree node
			childAddr, err := nr.ReadOffset()
			if err != nil {
				return nil, err
			}

			childEntries, err := readBTreeNode(r, childAddr, localHeap)
			if err != nil {
				return nil, err
			}
			entries = append(entries, childEntries...)
		}
	}

	return entries, nil
}

func readSymbolTableNode(r *binary.Reader, address uint64, localHeap *heap.LocalHeap) ([]GroupEntry, error) {
	nr := r.At(int64(address))

	// Check signature
	sig, err := nr.ReadBytes(4)
	if err != nil {
		return nil, fmt.Errorf("reading SNOD signature: %w", err)
	}
	if string(sig) != "SNOD" {
		return nil, fmt.Errorf("invalid symbol table node signature: got %q, expected \"SNOD\"", string(sig))
	}

	// Version (1 byte)
	version, err := nr.ReadUint8()
	if err != nil {
		return nil, err
	}
	if version != 1 {
		return nil, fmt.Errorf("unsupported symbol table node version: %d", version)
	}

	// Reserved (1 byte)
	nr.Skip(1)

	// Number of symbols (2 bytes)
	numSymbols, err := nr.ReadUint16()
	if err != nil {
		return nil, err
	}

	var entries []GroupEntry
	for i := uint16(0); i < numSymbols; i++ {
		entry, err := readSymbolTableEntry(nr, localHeap)
		if err != nil {
			return nil, fmt.Errorf("reading symbol table entry %d: %w", i, err)
		}
		if entry.Name != "" { // Skip empty entries
			entries = append(entries, entry)
		}
	}

	return entries, nil
}

// Symbol table entry cache types
const (
	cacheTypeNone     uint32 = 0 // No cached data
	cacheTypeHardLink uint32 = 1 // Object header info cached
	cacheTypeSoftLink uint32 = 2 // Symbolic link
)

func readSymbolTableEntry(r *binary.Reader, localHeap *heap.LocalHeap) (GroupEntry, error) {
	var entry GroupEntry

	// Link name offset (into local heap)
	nameOffset, err := r.ReadOffset()
	if err != nil {
		return entry, err
	}

	// Object header address
	objAddr, err := r.ReadOffset()
	if err != nil {
		return entry, err
	}

	// Cache type (4 bytes)
	cacheType, err := r.ReadUint32()
	if err != nil {
		return entry, err
	}

	// Reserved (4 bytes)
	r.Skip(4)

	// Scratch-pad space (16 bytes) - contents depend on cache type
	scratchPad, err := r.ReadBytes(16)
	if err != nil {
		return entry, err
	}

	// Get name from local heap
	entry.Name = localHeap.GetString(nameOffset)
	entry.ObjectAddress = objAddr
	entry.LinkType = 0 // Default to hard link

	switch cacheType {
	case cacheTypeNone, cacheTypeHardLink:
		// Hard link - object address is valid
		entry.LinkType = 0

	case cacheTypeSoftLink:
		// Soft link - scratch-pad contains offset to link value in local heap
		// The offset is stored as a 4-byte value at the start of scratch-pad
		linkOffset := uint64(scratchPad[0]) | uint64(scratchPad[1])<<8 |
			uint64(scratchPad[2])<<16 | uint64(scratchPad[3])<<24
		entry.LinkType = 1
		entry.SoftLinkValue = localHeap.GetString(linkOffset)
		entry.ObjectAddress = 0 // Not meaningful for soft links
	}

	return entry, nil
}
