// Package alloc provides space allocation management for HDF5 file writing.
//
// When writing HDF5 files, data structures like object headers, heaps, and
// chunk data must be placed at specific file offsets. This package manages
// the allocation of these offsets to prevent overlapping writes and track
// file growth.
//
// # Allocator
//
// The [Allocator] type provides thread-safe space management with the following
// features:
//
//   - Append-only allocation: New allocations are placed at the current
//     end-of-file address, which is then advanced.
//   - Aligned allocation: Allocations can be aligned to specific boundaries
//     (e.g., 8-byte alignment for object headers).
//   - Allocation tracking: All allocations are recorded for debugging and
//     validation purposes.
//   - Free space tracking: Freed blocks are tracked for potential future
//     space reuse (not yet implemented).
//
// # Usage
//
// Create an allocator with a base address (typically after the superblock):
//
//	alloc := alloc.New(96) // Start after 96-byte superblock
//	addr := alloc.Alloc(1024) // Allocate 1024 bytes
//	alignedAddr := alloc.AllocAligned(512, 8) // 8-byte aligned allocation
//
// The allocator can be converted to a simple function for compatibility:
//
//	allocFunc := alloc.AllocFunc()
//	addr := allocFunc(256)
package alloc
