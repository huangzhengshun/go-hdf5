package message

import (
	"fmt"

	binpkg "github.com/huangzhengshun/go-hdf5/internal/binary"
)

// SymbolTable represents a symbol table message (type 0x0011).
// This message is used in version 1 object headers to point to the
// B-tree and local heap that define group membership.
type SymbolTable struct {
	BTreeAddress    uint64 // Address of B-tree for group members
	LocalHeapAddress uint64 // Address of local heap for member names
}

func (m *SymbolTable) Type() Type { return TypeSymbolTable }

func parseSymbolTable(data []byte, r *binpkg.Reader) (*SymbolTable, error) {
	offsetSize := r.OffsetSize()

	if len(data) < 2*offsetSize {
		return nil, fmt.Errorf("symbol table message too short")
	}

	return &SymbolTable{
		BTreeAddress:     decodeUint(data[0:offsetSize], offsetSize, r.ByteOrder()),
		LocalHeapAddress: decodeUint(data[offsetSize:2*offsetSize], offsetSize, r.ByteOrder()),
	}, nil
}
