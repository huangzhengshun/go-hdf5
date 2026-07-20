package binary

import (
	"testing"
)

func TestLookup3Checksum(t *testing.T) {
	// Test consistency - same input should always produce same output
	tests := []struct {
		name  string
		input []byte
	}{
		{"empty", []byte{}},
		{"single byte", []byte{0x00}},
		{"hello", []byte("hello")},
		{"12 bytes exactly", []byte("Hello World!")},
		{"13 bytes", []byte("Hello World!!")},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result1 := Lookup3Checksum(tt.input)
			result2 := Lookup3Checksum(tt.input)
			if result1 != result2 {
				t.Errorf("Lookup3Checksum not consistent: got 0x%08x then 0x%08x",
					result1, result2)
			}
		})
	}
}

func TestLookup3ChecksumLengthVariations(t *testing.T) {
	// Test that different lengths produce different checksums
	checksums := make(map[uint32]int)

	for length := 0; length <= 24; length++ {
		data := make([]byte, length)
		for i := range data {
			data[i] = byte(i)
		}
		cs := Lookup3Checksum(data)
		checksums[cs] = length
	}

	// All 25 lengths should produce unique checksums
	if len(checksums) != 25 {
		t.Errorf("expected 25 unique checksums for lengths 0-24, got %d", len(checksums))
	}
}

func TestFletcher32(t *testing.T) {
	// Test basic properties and consistency
	tests := []struct {
		name  string
		input []byte
	}{
		{"empty", []byte{}},
		{"single byte", []byte{0x01}},
		{"two bytes", []byte{0x01, 0x02}},
		{"four bytes", []byte{0x01, 0x02, 0x03, 0x04}},
		{"hello", []byte("hello")},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result1 := Fletcher32(tt.input)
			result2 := Fletcher32(tt.input)
			if result1 != result2 {
				t.Errorf("Fletcher32 not consistent: got 0x%08x then 0x%08x",
					result1, result2)
			}
		})
	}

	// Empty input should produce zero
	if result := Fletcher32([]byte{}); result != 0 {
		t.Errorf("Fletcher32(empty) should be 0, got 0x%08x", result)
	}
}

func TestFletcher32OddLength(t *testing.T) {
	// Odd-length input should be handled correctly (zero-padded)
	odd := []byte{0x01, 0x02, 0x03}
	even := []byte{0x01, 0x02, 0x03, 0x00}

	oddResult := Fletcher32(odd)
	evenResult := Fletcher32(even)

	if oddResult != evenResult {
		t.Errorf("Fletcher32 should pad odd-length input: odd=0x%08x, even=0x%08x",
			oddResult, evenResult)
	}
}

func TestVerifyFletcher32(t *testing.T) {
	data := []byte("test data for verification")
	checksum := Fletcher32(data)

	if !VerifyFletcher32(data, checksum) {
		t.Error("VerifyFletcher32 should return true for matching checksum")
	}

	if VerifyFletcher32(data, checksum+1) {
		t.Error("VerifyFletcher32 should return false for non-matching checksum")
	}
}

func TestVerifyLookup3(t *testing.T) {
	data := []byte("test data for verification")
	checksum := Lookup3Checksum(data)

	if !VerifyLookup3(data, checksum) {
		t.Error("VerifyLookup3 should return true for matching checksum")
	}

	if VerifyLookup3(data, checksum+1) {
		t.Error("VerifyLookup3 should return false for non-matching checksum")
	}
}

func BenchmarkLookup3Checksum(b *testing.B) {
	data := make([]byte, 4096)
	for i := range data {
		data[i] = byte(i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Lookup3Checksum(data)
	}
}

func BenchmarkFletcher32(b *testing.B) {
	data := make([]byte, 4096)
	for i := range data {
		data[i] = byte(i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Fletcher32(data)
	}
}
