package message

import (
	"fmt"

	binpkg "github.com/huangzhengshun/go-hdf5/internal/binary"
)

// FillValueStatus indicates when fill values are written.
type FillValueStatus uint8

const (
	FillUndefined   FillValueStatus = 0
	FillDefault     FillValueStatus = 1
	FillUserDefined FillValueStatus = 2
)

// FillValue represents a fill value message (type 0x0005).
type FillValue struct {
	Version      uint8
	SpaceAllocTime uint8
	FillWriteTime  uint8
	IsDefined    bool
	Size         uint32
	Value        []byte
}

func (m *FillValue) Type() Type { return TypeFillValue }

func parseFillValue(data []byte, r *binpkg.Reader) (*FillValue, error) {
	if len(data) < 2 {
		return nil, fmt.Errorf("fill value message too short")
	}

	fv := &FillValue{
		Version: data[0],
	}

	switch fv.Version {
	case 1, 2:
		return parseFillValueV1V2(data, fv)
	case 3:
		return parseFillValueV3(data, fv)
	default:
		return nil, fmt.Errorf("unsupported fill value version: %d", fv.Version)
	}
}

func parseFillValueV1V2(data []byte, fv *FillValue) (*FillValue, error) {
	if len(data) < 4 {
		return nil, fmt.Errorf("fill value v1/v2 too short")
	}

	fv.SpaceAllocTime = data[1]
	fv.FillWriteTime = data[2]
	fv.IsDefined = data[3] != 0

	if fv.IsDefined && len(data) >= 8 {
		fv.Size = uint32(data[4]) | uint32(data[5])<<8 |
			uint32(data[6])<<16 | uint32(data[7])<<24
		if len(data) >= 8+int(fv.Size) {
			fv.Value = make([]byte, fv.Size)
			copy(fv.Value, data[8:8+fv.Size])
		}
	}

	return fv, nil
}

func parseFillValueV3(data []byte, fv *FillValue) (*FillValue, error) {
	if len(data) < 2 {
		return nil, fmt.Errorf("fill value v3 too short")
	}

	flags := data[1]
	fv.SpaceAllocTime = flags & 0x03
	fv.FillWriteTime = (flags >> 2) & 0x03
	fv.IsDefined = (flags>>4)&0x01 == 0 // Bit 4: undefined flag (0 = defined)

	offset := 2
	if fv.IsDefined && (flags>>5)&0x01 != 0 {
		// Fill value is present
		if offset+4 > len(data) {
			return nil, fmt.Errorf("fill value v3 size truncated")
		}
		fv.Size = uint32(data[offset]) | uint32(data[offset+1])<<8 |
			uint32(data[offset+2])<<16 | uint32(data[offset+3])<<24
		offset += 4

		if offset+int(fv.Size) > len(data) {
			return nil, fmt.Errorf("fill value v3 data truncated")
		}
		fv.Value = make([]byte, fv.Size)
		copy(fv.Value, data[offset:offset+int(fv.Size)])
	}

	return fv, nil
}
