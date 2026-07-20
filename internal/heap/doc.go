// Package heap implements HDF5 heap structures for storing variable-length data.
//
// HDF5 uses two types of heaps to store variable-length data like object names
// and variable-length string values:
//
// # Local Heap
//
// The [LocalHeap] (signature "HEAP") stores variable-length data for v0/v1
// groups, primarily object names. Each v0/v1 group has an associated local
// heap where member names are stored as null-terminated strings.
//
// Local heap structure:
//   - Fixed header with data segment size and free list offset
//   - Data segment containing null-terminated strings
//   - Symbol table entries reference strings by offset into this heap
//
// Usage:
//
//	heap, err := heap.ReadLocalHeap(reader, heapAddress)
//	name := heap.GetString(nameOffset)
//
// # Global Heap
//
// The [GlobalHeap] (signature "GCOL") stores variable-length data that may be
// shared across multiple objects, such as variable-length string values and
// variable-length sequence data. Global heap collections contain numbered
// objects that can be referenced by a (collection address, object index) pair.
//
// Global heap structure:
//   - Collection header with total size
//   - Numbered objects with reference counts
//   - Objects are padded to 8-byte boundaries
//
// Usage:
//
//	heap, err := heap.ReadGlobalHeap(reader, collectionAddress)
//	data, err := heap.GetObject(objectIndex)
//	str, err := heap.GetString(objectIndex)
//
// # Global Heap ID
//
// Variable-length data fields in datasets store a [GlobalHeapID] which
// contains the collection address and object index needed to retrieve
// the actual data:
//
//	heapID, err := heap.ParseGlobalHeapID(rawBytes, offsetSize)
//	heap, err := heap.ReadGlobalHeap(reader, heapID.CollectionAddress)
//	value, err := heap.GetObject(uint16(heapID.ObjectIndex))
//
// # Key Types
//
//   - [LocalHeap]: Local heap for group names (v0/v1 groups)
//   - [GlobalHeap]: Global heap collection for variable-length data
//   - [GlobalHeapID]: Reference to an object in a global heap
package heap
