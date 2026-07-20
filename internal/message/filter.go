package message

import (
	"encoding/binary"
	"fmt"

	binpkg "github.com/huangzhengshun/go-hdf5/internal/binary"
)

// Filter IDs
const (
	FilterDeflate     uint16 = 1 // DEFLATE (gzip)
	FilterShuffle     uint16 = 2 // Byte shuffle
	FilterFletcher32  uint16 = 3 // Fletcher32 checksum
	FilterSZIP        uint16 = 4 // SZIP compression
	FilterNBit        uint16 = 5 // N-bit packing
	FilterScaleOffset uint16 = 6 // Scale + offset
)

// FilterInfo describes a single filter in the pipeline.
type FilterInfo struct {
	ID         uint16   // Filter identifier
	Flags      uint16   // Filter flags (bit 0: optional)
	Name       string   // Filter name (optional, v1 only)
	ClientData []uint32 // Filter parameters
}

// IsOptional returns true if this filter is optional.
func (f *FilterInfo) IsOptional() bool {
	return f.Flags&0x01 != 0
}

// FilterPipeline represents a filter pipeline message (type 0x000B).
type FilterPipeline struct {
	Version uint8
	Filters []FilterInfo
}

func (m *FilterPipeline) Type() Type { return TypeFilterPipeline }

// HasFilter returns true if the pipeline contains the given filter ID.
func (m *FilterPipeline) HasFilter(id uint16) bool {
	for _, f := range m.Filters {
		if f.ID == id {
			return true
		}
	}
	return false
}

// HasCompression returns true if the pipeline has any compression filter.
func (m *FilterPipeline) HasCompression() bool {
	for _, f := range m.Filters {
		switch f.ID {
		case FilterDeflate, FilterSZIP:
			return true
		}
	}
	return false
}

func parseFilterPipeline(data []byte, r *binpkg.Reader) (*FilterPipeline, error) {
	if len(data) < 2 {
		return nil, fmt.Errorf("filter pipeline message too short")
	}

	fp := &FilterPipeline{
		Version: data[0],
		Filters: make([]FilterInfo, data[1]),
	}

	offset := 2

	// Version 1 has 6 reserved bytes
	if fp.Version == 1 {
		offset = 8
	}

	for i := range fp.Filters {
		filter, consumed, err := parseFilterInfo(data[offset:], fp.Version)
		if err != nil {
			return nil, fmt.Errorf("parsing filter %d: %w", i, err)
		}
		fp.Filters[i] = filter
		offset += consumed
	}

	return fp, nil
}

func parseFilterInfo(data []byte, version uint8) (FilterInfo, int, error) {
	var f FilterInfo

	if len(data) < 6 {
		return f, 0, fmt.Errorf("filter info too short")
	}

	f.ID = binary.LittleEndian.Uint16(data[0:2])
	offset := 2

	// Name length field only present in v1 or for custom filters (ID >= 256)
	var nameLen uint16
	if version == 1 || f.ID >= 256 {
		nameLen = binary.LittleEndian.Uint16(data[offset:])
		offset += 2
	}

	f.Flags = binary.LittleEndian.Uint16(data[offset:])
	offset += 2
	numCD := binary.LittleEndian.Uint16(data[offset:])
	offset += 2

	// Parse name (v1 only, or custom filters)
	if nameLen > 0 {
		if offset+int(nameLen) > len(data) {
			return f, 0, fmt.Errorf("filter name truncated")
		}
		// Find null terminator
		nameEnd := offset
		for nameEnd < offset+int(nameLen) && data[nameEnd] != 0 {
			nameEnd++
		}
		f.Name = string(data[offset:nameEnd])
		offset += int(nameLen)

		// v1: names are padded to 8-byte boundary
		if version == 1 && nameLen%8 != 0 {
			offset += 8 - int(nameLen%8)
		}
	}

	// Parse client data
	f.ClientData = make([]uint32, numCD)
	for j := 0; j < int(numCD) && offset+4 <= len(data); j++ {
		f.ClientData[j] = binary.LittleEndian.Uint32(data[offset:])
		offset += 4
	}

	// v1: padding if odd number of client data values
	if version == 1 && numCD%2 != 0 {
		offset += 4
	}

	return f, offset, nil
}
