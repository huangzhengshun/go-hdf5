package alloc

import (
	"testing"
)

func TestAllocatorBasic(t *testing.T) {
	a := New(1024) // Start at 1KB

	// First allocation
	addr1 := a.Alloc(100)
	if addr1 != 1024 {
		t.Errorf("first allocation: got 0x%x, want 0x%x", addr1, 1024)
	}

	// Second allocation
	addr2 := a.Alloc(200)
	if addr2 != 1124 {
		t.Errorf("second allocation: got 0x%x, want 0x%x", addr2, 1124)
	}

	// Check EOF
	if a.EOFAddr() != 1324 {
		t.Errorf("EOF: got 0x%x, want 0x%x", a.EOFAddr(), 1324)
	}
}

func TestAllocatorZeroSize(t *testing.T) {
	a := New(100)

	addr := a.Alloc(0)
	if addr != 100 {
		t.Errorf("zero allocation: got 0x%x, want 0x%x", addr, 100)
	}

	// EOF should not change
	if a.EOFAddr() != 100 {
		t.Errorf("EOF after zero alloc: got 0x%x, want 0x%x", a.EOFAddr(), 100)
	}
}

func TestAllocatorAligned(t *testing.T) {
	a := New(100)

	// Allocate something first to misalign
	a.Alloc(13) // Now at 113

	// Allocate with 8-byte alignment
	addr := a.AllocAligned(50, 8)
	if addr%8 != 0 {
		t.Errorf("aligned allocation not aligned: 0x%x %% 8 = %d", addr, addr%8)
	}
	if addr != 120 { // 113 -> 120 (aligned)
		t.Errorf("aligned allocation: got 0x%x, want 0x%x", addr, 120)
	}
}

func TestAllocatorStats(t *testing.T) {
	a := New(0)

	a.Alloc(100)
	a.Alloc(200)
	a.Alloc(50)

	stats := a.Stats()
	if stats.TotalAllocations != 3 {
		t.Errorf("TotalAllocations: got %d, want 3", stats.TotalAllocations)
	}
	if stats.TotalBytesAlloc != 350 {
		t.Errorf("TotalBytesAlloc: got %d, want 350", stats.TotalBytesAlloc)
	}
	if stats.LargestAlloc != 200 {
		t.Errorf("LargestAlloc: got %d, want 200", stats.LargestAlloc)
	}
}

func TestAllocatorValidate(t *testing.T) {
	a := New(100)

	a.Alloc(50)
	a.Alloc(100)
	a.Alloc(75)

	if err := a.Validate(); err != nil {
		t.Errorf("valid allocations should not error: %v", err)
	}
}

func TestAllocatorAllocFunc(t *testing.T) {
	a := New(0)
	allocFunc := a.AllocFunc()

	addr1 := allocFunc(100)
	addr2 := allocFunc(200)

	if addr1 != 0 {
		t.Errorf("first: got 0x%x, want 0", addr1)
	}
	if addr2 != 100 {
		t.Errorf("second: got 0x%x, want 100", addr2)
	}
}

func TestAllocatorReset(t *testing.T) {
	a := New(1000)

	a.Alloc(100)
	a.Alloc(200)

	a.Reset()

	if a.EOFAddr() != 1000 {
		t.Errorf("EOF after reset: got 0x%x, want 0x%x", a.EOFAddr(), 1000)
	}
	if len(a.Allocations()) != 0 {
		t.Errorf("allocations after reset: got %d, want 0", len(a.Allocations()))
	}
}

func TestAllocatorFree(t *testing.T) {
	a := New(0)

	addr := a.Alloc(100)
	a.Free(addr, 100)

	stats := a.Stats()
	if stats.TotalBytesFree != 100 {
		t.Errorf("TotalBytesFree: got %d, want 100", stats.TotalBytesFree)
	}

	freeBlocks := a.FreeBlocks()
	if len(freeBlocks) != 1 {
		t.Errorf("FreeBlocks: got %d, want 1", len(freeBlocks))
	}
}

func TestAllocatorTagged(t *testing.T) {
	a := New(0)

	a.AllocTagged(100, "root_group")
	a.AllocTagged(200, "dataset")

	allocs := a.Allocations()
	if len(allocs) != 2 {
		t.Fatalf("allocations: got %d, want 2", len(allocs))
	}
	if allocs[0].Tag != "root_group" {
		t.Errorf("first tag: got %q, want %q", allocs[0].Tag, "root_group")
	}
	if allocs[1].Tag != "dataset" {
		t.Errorf("second tag: got %q, want %q", allocs[1].Tag, "dataset")
	}
}
