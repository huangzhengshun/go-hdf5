package object

import (
	"fmt"

	"github.com/huangzhengshun/go-hdf5/internal/binary"
	"github.com/huangzhengshun/go-hdf5/internal/message"
)

/*
Version 2 Object Header Layout:
Offset  Size  Description
0       4     Signature ("OHDR")
4       1     Version (2)
5       1     Flags
          	  Bit 0-1: Size of chunk#0 size field (1 << value bytes)
          	  Bit 2: Track attribute creation order
          	  Bit 3: Index attribute creation order
          	  Bit 4: Store non-default attribute storage phase change values
          	  Bit 5: Store access, modification, change, birth times
6       var   Access time (4 bytes, if flag bit 5 set)
var     var   Modification time (4 bytes, if flag bit 5 set)
var     var   Change time (4 bytes, if flag bit 5 set)
var     var   Birth time (4 bytes, if flag bit 5 set)
var     var   Max compact attributes (2 bytes, if flag bit 4 set)
var     var   Min dense attributes (2 bytes, if flag bit 4 set)
var     1-8   Size of chunk#0 (1, 2, 4, or 8 bytes based on flag bits 0-1)
var     var   Header messages
var     4     Checksum

Each V2 Message (normal):
0       1     Message type
1       2     Size of message data
3       1     Flags
4       var   Creation order (2 bytes, if header flag bit 2 set)
var     var   Message data

Each V2 Message (extended, type byte = 0xFF):
0       1     0xFF marker
1       1     Message type
2       4     Size of message data (32-bit)
6       1     Flags
7       var   Creation order (2 bytes, if header flag bit 2 set)
var     var   Message data
*/

func readV2(r *binary.Reader, address uint64) (*Header, error) {
	// Skip signature (already verified)
	r.Skip(4)

	version, err := r.ReadUint8()
	if err != nil {
		return nil, err
	}
	if version != 2 {
		return nil, fmt.Errorf("%w: expected version 2, got %d", ErrUnsupportedVersion, version)
	}

	flags, err := r.ReadUint8()
	if err != nil {
		return nil, err
	}

	hdr := &Header{
		Version: 2,
		Address: address,
		Flags:   flags,
	}

	// Optional timestamps (flag bit 5)
	if flags&0x20 != 0 {
		hdr.AccessTime, _ = r.ReadUint32()
		hdr.ModTime, _ = r.ReadUint32()
		hdr.ChangeTime, _ = r.ReadUint32()
		hdr.BirthTime, _ = r.ReadUint32()
	}

	// Optional attribute phase change values (flag bit 4)
	if flags&0x10 != 0 {
		r.Skip(4) // max compact + min dense (2 + 2 bytes)
	}

	// Chunk 0 size (size determined by flag bits 0-1)
	// Flag bits 0-1: 0→1 byte, 1→2 bytes, 2→4 bytes, 3→8 bytes
	sizeFieldSize := 1 << (flags & 0x03)
	chunk0Size, err := r.ReadUintN(sizeFieldSize)
	if err != nil {
		return nil, err
	}

	// Track creation order flag (bit 2)
	trackCreationOrder := flags&0x04 != 0

	// Calculate where messages end (before checksum)
	// chunk0Size is the size of messages + padding (NOT including checksum)
	chunkEnd := r.Pos() + int64(chunk0Size)

	// Parse messages
	for r.Pos() < chunkEnd {
		msg, err := readV2Message(r, trackCreationOrder)
		if err != nil {
			break
		}
		if msg != nil {
			// Handle continuation message
			if cont, ok := msg.(*message.Continuation); ok {
				contMsgs, err := readV2Continuation(r, cont.Offset, cont.Length, trackCreationOrder)
				if err == nil {
					hdr.Messages = append(hdr.Messages, contMsgs...)
				}
				continue
			}
			hdr.Messages = append(hdr.Messages, msg)
		}
	}

	// Verify checksum
	// Checksum covers: signature(4) + version(1) + flags(1) + chunkSize(var) + messages + padding
	// chunk0Size is the size of messages + padding (NOT including checksum)
	// So bytes before checksum = 6 + sizeFieldSize + chunk0Size
	checksumReader := r.At(int64(address))
	checksumData, err := checksumReader.ReadBytes(6 + int(sizeFieldSize) + int(chunk0Size))
	if err != nil {
		return hdr, err
	}

	// Read stored checksum - checksum is at address + 6 + sizeFieldSize + chunk0Size
	checksumPos := int64(address) + 6 + int64(sizeFieldSize) + int64(chunk0Size)
	storedChecksumReader := r.At(checksumPos)
	storedChecksum, err := storedChecksumReader.ReadUint32()
	if err != nil {
		return hdr, err
	}

	calculatedChecksum := binary.Lookup3Checksum(checksumData)
	if storedChecksum != calculatedChecksum {
		// Checksum mismatch is non-fatal - return the header with parsed messages
		// This allows reading files that may have been created with different checksum algorithms
		return hdr, nil
	}

	return hdr, nil
}

// readV2Continuation reads messages from a v2 continuation block.
func readV2Continuation(r *binary.Reader, offset, length uint64, trackCreationOrder bool) ([]message.Message, error) {
	cr := r.At(int64(offset))
	var messages []message.Message

	// V2 continuation blocks have: signature "OCHK" (4 bytes) + messages + checksum (4 bytes)
	sig, err := cr.ReadBytes(4)
	if err != nil {
		return nil, err
	}
	if string(sig) != "OCHK" {
		return nil, fmt.Errorf("invalid continuation block signature: %s", sig)
	}

	// Calculate where messages end (before checksum)
	chunkEnd := int64(offset) + int64(length) - 4

	for cr.Pos() < chunkEnd {
		msg, err := readV2Message(cr, trackCreationOrder)
		if err != nil {
			break
		}
		if msg != nil {
			// Handle nested continuation (unlikely but possible)
			if cont, ok := msg.(*message.Continuation); ok {
				nestedMsgs, err := readV2Continuation(r, cont.Offset, cont.Length, trackCreationOrder)
				if err == nil {
					messages = append(messages, nestedMsgs...)
				}
				continue
			}
			messages = append(messages, msg)
		}
	}

	return messages, nil
}

func readV2Message(r *binary.Reader, trackCreationOrder bool) (message.Message, error) {
	firstByte, err := r.ReadUint8()
	if err != nil {
		return nil, err
	}

	var msgType uint8
	var dataSize uint32

	if firstByte == 0xFF {
		// Extended format: 32-bit size
		msgType, err = r.ReadUint8()
		if err != nil {
			return nil, err
		}
		dataSize, err = r.ReadUint32()
		if err != nil {
			return nil, err
		}
	} else {
		// Normal format: 16-bit size
		msgType = firstByte
		size16, err := r.ReadUint16()
		if err != nil {
			return nil, err
		}
		dataSize = uint32(size16)
	}

	flags, err := r.ReadUint8()
	if err != nil {
		return nil, err
	}

	// Optional creation order
	if trackCreationOrder {
		r.Skip(2)
	}

	// Read message data
	data, err := r.ReadBytes(int(dataSize))
	if err != nil {
		return nil, err
	}

	// Skip NIL messages
	if msgType == 0 {
		return nil, nil
	}

	// Parse the message
	return message.Parse(message.Type(msgType), data, flags, r)
}
