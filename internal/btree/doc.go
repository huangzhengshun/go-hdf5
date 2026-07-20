// Package btree implements HDF5 B-tree structures for group and chunk indexing.
//
// HDF5 uses B-trees to efficiently index both group members (in older v0/v1
// superblock files) and chunked dataset storage. This package provides readers
// for both B-tree versions used in HDF5 files.
//
// # B-tree Versions
//
// HDF5 defines two B-tree versions:
//
//   - V1 B-trees (signature "TREE"): Used in older files for group symbol
//     tables and chunked storage. Group B-trees point to Symbol Table Nodes
//     (SNOD) which contain the actual group entries.
//
//   - V2 B-trees (signature "BTHD"): Modern format with better performance
//     characteristics. Used for chunked storage in newer files with types
//     10 (without filter info) and 11 (with filter info).
//
// # Group Indexing
//
// For v0/v1 superblock files, groups use a B-tree + local heap combination:
//
//   - [ReadGroupEntries] traverses a v1 B-tree to find all group members
//   - Each B-tree leaf points to Symbol Table Nodes containing entries
//   - Entry names are stored in the associated [heap.LocalHeap]
//
// # Chunk Indexing
//
// Chunked datasets store their data in separate chunks, indexed by B-trees:
//
//   - [ReadChunkIndex] reads a v1 B-tree chunk index
//   - [ReadChunkIndexV2] reads a v2 B-tree chunk index
//   - [ChunkEntry] contains the chunk offset, address, size, and filter mask
//   - [ChunkIndex] provides a FindChunk method for coordinate-based lookup
//
// # Key Types
//
//   - [ChunkEntry]: Represents a single chunk with its file address and metadata
//   - [ChunkIndex]: Collection of chunk entries with lookup capability
//   - [GroupEntry]: Represents a group member (name, address, link type)
package btree
