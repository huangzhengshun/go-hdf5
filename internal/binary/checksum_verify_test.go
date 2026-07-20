package binary

import (
    "testing"
    "os"
    "encoding/binary"
)

func TestVerifyMinimalChecksum(t *testing.T) {
    data, err := os.ReadFile("/tmp/minimal2.h5")
    if err != nil {
        t.Skip("Test file not found")
    }
    
    // Check superblock checksum
    t.Log("=== Superblock Checksum ===")
    sbData := data[:44]
    sbChecksum := binary.LittleEndian.Uint32(data[44:48])
    calculatedSb := Lookup3Checksum(sbData)
    t.Logf("Stored: 0x%08x", sbChecksum)
    t.Logf("Calculated: 0x%08x", calculatedSb)
    if sbChecksum != calculatedSb {
        t.Error("Superblock checksum mismatch")
    }
    
    // Check OHDR checksum at 0x30
    // Note: chunk size does NOT include the 4-byte checksum
    t.Log("=== OHDR Checksum ===")
    ohdrStart := 0x30
    chunkSize := int(data[ohdrStart+6])
    ohdrSize := 7 + chunkSize + 4 // prefix + chunk + checksum

    t.Logf("OHDR at 0x%x, chunk size %d, total size %d", ohdrStart, chunkSize, ohdrSize)

    checksumOffset := ohdrStart + 7 + chunkSize // checksum is at end of chunk
    ohdrData := data[ohdrStart:checksumOffset]
    storedChecksum := binary.LittleEndian.Uint32(data[checksumOffset:])
    calculatedOhdr := Lookup3Checksum(ohdrData)
    
    t.Logf("Checksum at 0x%x", checksumOffset)
    t.Logf("Stored: 0x%08x", storedChecksum)
    t.Logf("Calculated: 0x%08x", calculatedOhdr)
    if storedChecksum != calculatedOhdr {
        t.Error("OHDR checksum mismatch")
    }
}
