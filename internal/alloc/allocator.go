// Package alloc provides space management for HDF5 file writing.
package alloc

import (
	"fmt"
	"sync"
)

// Allocator manages space allocation within an HDF5 file.
// It provides append-only allocation by default, with optional
// free space tracking for future space reuse.
type Allocator struct {
	mu sync.Mutex

	// eofAddr is the current end-of-file address (next allocation point)
	eofAddr uint64

	// baseAddr is the minimum address that can be allocated
	// (typically after the superblock)
	baseAddr uint64

	// allocations tracks all allocations made (for debugging/validation)
	allocations []Allocation

	// freeBlocks tracks freed space (for future space reuse)
	freeBlocks []FreeBlock

	// stats tracks allocation statistics
	stats Stats
}

// Allocation represents a single allocation made.
type Allocation struct {
	Addr uint64
	Size uint64
	Tag  string // Optional tag for debugging
}

// FreeBlock represents a freed block of space.
type FreeBlock struct {
	Addr uint64
	Size uint64
}

// Stats contains allocation statistics.
type Stats struct {
	TotalAllocations uint64 // Number of allocations made
	TotalBytesAlloc  uint64 // Total bytes allocated
	TotalBytesFree   uint64 // Total bytes freed (for future reuse)
	LargestAlloc     uint64 // Largest single allocation
}

// New creates a new Allocator starting at the given base address.
// The base address is typically right after the superblock.
func New(baseAddr uint64) *Allocator {
	return &Allocator{
		eofAddr:  baseAddr,
		baseAddr: baseAddr,
	}
}

// Alloc allocates a block of the given size and returns its address.
// This is a simple append-only allocation that always allocates at EOF.
func (a *Allocator) Alloc(size uint64) uint64 {
	a.mu.Lock()
	defer a.mu.Unlock()

	return a.allocLocked(size, "")
}

// AllocTagged allocates a block with an optional tag for debugging.
func (a *Allocator) AllocTagged(size uint64, tag string) uint64 {
	a.mu.Lock()
	defer a.mu.Unlock()

	return a.allocLocked(size, tag)
}

// allocLocked performs allocation while holding the lock.
func (a *Allocator) allocLocked(size uint64, tag string) uint64 {
	if size == 0 {
		return a.eofAddr
	}

	addr := a.eofAddr
	a.eofAddr += size

	// Track allocation
	a.allocations = append(a.allocations, Allocation{
		Addr: addr,
		Size: size,
		Tag:  tag,
	})

	// Update stats
	a.stats.TotalAllocations++
	a.stats.TotalBytesAlloc += size
	if size > a.stats.LargestAlloc {
		a.stats.LargestAlloc = size
	}

	return addr
}

// AllocAligned allocates a block with the given alignment.
// The returned address will be aligned to the specified boundary.
func (a *Allocator) AllocAligned(size uint64, alignment uint64) uint64 {
	a.mu.Lock()
	defer a.mu.Unlock()

	if alignment <= 1 {
		return a.allocLocked(size, "")
	}

	// Calculate padding needed for alignment
	remainder := a.eofAddr % alignment
	if remainder != 0 {
		padding := alignment - remainder
		a.eofAddr += padding
	}

	return a.allocLocked(size, "")
}

// Free marks a block as free for future reuse.
// Note: This is currently just tracking; actual reuse is not yet implemented.
func (a *Allocator) Free(addr, size uint64) {
	a.mu.Lock()
	defer a.mu.Unlock()

	a.freeBlocks = append(a.freeBlocks, FreeBlock{
		Addr: addr,
		Size: size,
	})
	a.stats.TotalBytesFree += size
}

// EOFAddr returns the current end-of-file address.
func (a *Allocator) EOFAddr() uint64 {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.eofAddr
}

// SetEOFAddr sets the EOF address (used when loading existing files).
func (a *Allocator) SetEOFAddr(addr uint64) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.eofAddr = addr
}

// BaseAddr returns the base address (start of allocatable space).
func (a *Allocator) BaseAddr() uint64 {
	return a.baseAddr
}

// Stats returns a copy of the allocation statistics.
func (a *Allocator) Stats() Stats {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.stats
}

// Allocations returns a copy of all allocations made (for debugging).
func (a *Allocator) Allocations() []Allocation {
	a.mu.Lock()
	defer a.mu.Unlock()
	result := make([]Allocation, len(a.allocations))
	copy(result, a.allocations)
	return result
}

// FreeBlocks returns a copy of all free blocks (for debugging).
func (a *Allocator) FreeBlocks() []FreeBlock {
	a.mu.Lock()
	defer a.mu.Unlock()
	result := make([]FreeBlock, len(a.freeBlocks))
	copy(result, a.freeBlocks)
	return result
}

// Validate checks that allocations don't overlap and are within bounds.
func (a *Allocator) Validate() error {
	a.mu.Lock()
	defer a.mu.Unlock()

	if len(a.allocations) == 0 {
		return nil
	}

	// Check each allocation is within bounds
	for _, alloc := range a.allocations {
		if alloc.Addr < a.baseAddr {
			return fmt.Errorf("allocation at 0x%x is before base address 0x%x", alloc.Addr, a.baseAddr)
		}
		if alloc.Addr+alloc.Size > a.eofAddr {
			return fmt.Errorf("allocation at 0x%x size %d extends past EOF 0x%x", alloc.Addr, alloc.Size, a.eofAddr)
		}
	}

	// Check for overlaps (simple O(n²) check for debugging)
	for i := 0; i < len(a.allocations); i++ {
		for j := i + 1; j < len(a.allocations); j++ {
			a1, a2 := a.allocations[i], a.allocations[j]
			// Check if they overlap
			if a1.Addr < a2.Addr+a2.Size && a2.Addr < a1.Addr+a1.Size {
				return fmt.Errorf("overlapping allocations: [0x%x, size %d] and [0x%x, size %d]",
					a1.Addr, a1.Size, a2.Addr, a2.Size)
			}
		}
	}

	return nil
}

// AllocFunc returns an allocation function compatible with existing code.
// This allows gradual migration from the simple closure-based allocator.
// If a negative size is passed, the function returns 0 (safe no-op).
func (a *Allocator) AllocFunc() func(size int64) uint64 {
	return func(size int64) uint64 {
		if size <= 0 {
			return 0
		}
		return a.Alloc(uint64(size))
	}
}

// Reset resets the allocator to its initial state.
// This is primarily useful for testing.
func (a *Allocator) Reset() {
	a.mu.Lock()
	defer a.mu.Unlock()

	a.eofAddr = a.baseAddr
	a.allocations = nil
	a.freeBlocks = nil
	a.stats = Stats{}
}
