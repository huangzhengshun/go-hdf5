package object

import (
	"fmt"

	"github.com/huangzhengshun/go-hdf5/internal/binary"
	"github.com/huangzhengshun/go-hdf5/internal/message"
)

/*
Version 1 Object Header Layout:
Offset  Size  Description
0       1     Version (1)
1       1     Reserved
2       2     Number of header messages
4       4     Object reference count
8       4     Object header size (bytes of messages)
12      var   Header messages (8-byte aligned)

Each V1 Message:
0       2     Message type
2       2     Size of message data
4       1     Flags
5       3     Reserved
8       var   Message data
        pad   Padding to 8-byte boundary
*/

func readV1(r *binary.Reader, address uint64) (*Header, error) {
	version, err := r.ReadUint8()
	if err != nil {
		return nil, err
	}
	if version != 1 {
		return nil, fmt.Errorf("%w: expected version 1, got %d", ErrUnsupportedVersion, version)
	}

	r.Skip(1) // Reserved

	numMessages, err := r.ReadUint16()
	if err != nil {
		return nil, err
	}

	refCount, err := r.ReadUint32()
	if err != nil {
		return nil, err
	}

	headerSize, err := r.ReadUint32()
	if err != nil {
		return nil, err
	}

	hdr := &Header{
		Version:  1,
		Address:  address,
		RefCount: refCount,
		Messages: make([]message.Message, 0, numMessages),
	}

	// Align to 8-byte boundary before reading messages
	r.Align(8)

	messagesStart := r.Pos()
	messagesEnd := messagesStart + int64(headerSize)

	for r.Pos() < messagesEnd {
		msgType, err := r.ReadUint16()
		if err != nil {
			break
		}

		dataSize, err := r.ReadUint16()
		if err != nil {
			break
		}

		flags, err := r.ReadUint8()
		if err != nil {
			break
		}

		r.Skip(3) // Reserved

		data, err := r.ReadBytes(int(dataSize))
		if err != nil {
			break
		}

		// Align to 8-byte boundary
		r.Align(8)

		// Skip NIL messages (type 0)
		if msgType == 0 {
			continue
		}

		// Handle continuation message
		if message.Type(msgType) == message.TypeObjectHeaderContinuation {
			contMsg, err := message.ParseContinuation(data, r)
			if err != nil {
				continue
			}
			// Read messages from continuation block
			contMsgs, err := readV1Continuation(r, contMsg.Offset, contMsg.Length)
			if err == nil {
				hdr.Messages = append(hdr.Messages, contMsgs...)
			}
			continue
		}

		// Parse the message
		msg, err := message.Parse(message.Type(msgType), data, flags, r)
		if err != nil {
			// Skip unknown message types
			continue
		}

		hdr.Messages = append(hdr.Messages, msg)
	}

	return hdr, nil
}

func readV1Continuation(r *binary.Reader, offset, length uint64) ([]message.Message, error) {
	cr := r.At(int64(offset))
	var messages []message.Message

	endPos := int64(offset + length)
	for cr.Pos() < endPos {
		msgType, err := cr.ReadUint16()
		if err != nil {
			break
		}

		dataSize, err := cr.ReadUint16()
		if err != nil {
			break
		}

		flags, err := cr.ReadUint8()
		if err != nil {
			break
		}

		cr.Skip(3) // Reserved

		data, err := cr.ReadBytes(int(dataSize))
		if err != nil {
			break
		}

		cr.Align(8)

		if msgType == 0 {
			continue
		}

		// Handle nested continuation messages recursively
		if message.Type(msgType) == message.TypeObjectHeaderContinuation {
			contMsg, err := message.ParseContinuation(data, cr)
			if err != nil {
				continue
			}
			nestedMsgs, err := readV1Continuation(cr, contMsg.Offset, contMsg.Length)
			if err == nil {
				messages = append(messages, nestedMsgs...)
			}
			continue
		}

		msg, err := message.Parse(message.Type(msgType), data, flags, cr)
		if err != nil {
			continue
		}

		messages = append(messages, msg)
	}

	return messages, nil
}
