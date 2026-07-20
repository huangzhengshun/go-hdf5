// Package message handles parsing of HDF5 object header messages.
//
// Object headers contain a sequence of messages that describe the properties
// of HDF5 objects (groups, datasets, committed datatypes). Each message has
// a type, flags, and type-specific content.
//
// # Message Types
//
// This package defines constants for all HDF5 message types and implements
// parsing for the following:
//
//   - Dataspace (0x0001): Describes the dimensions of a dataset. See [Dataspace].
//   - Datatype (0x0003): Describes the data type of elements. See [Datatype].
//   - Fill Value (0x0005): Specifies the fill value for unwritten data.
//   - Link (0x0006): Describes a link to another object. See [Link].
//   - Data Layout (0x0008): Describes how dataset data is stored. See [DataLayout].
//   - Filter Pipeline (0x000B): Lists filters applied to chunks. See [FilterPipeline].
//   - Attribute (0x000C): Stores an attribute name, datatype, and value. See [Attribute].
//   - Symbol Table (0x0011): Points to v1 group B-tree and heap. See [SymbolTable].
//   - Continuation (0x0010): Points to additional header data. See [Continuation].
//
// Unrecognized message types are wrapped in [Unknown] for forward compatibility.
//
// # Datatype Classes
//
// The [Datatype] message supports the following HDF5 type classes:
//
//   - ClassFixedPoint (0): Integers (signed/unsigned, various sizes)
//   - ClassFloatPoint (1): IEEE floating-point numbers
//   - ClassString (3): Fixed-length strings
//   - ClassBitfield (4): Bit fields
//   - ClassOpaque (5): Opaque byte sequences
//   - ClassCompound (6): Structures with named members
//   - ClassReference (7): Object or region references
//   - ClassEnum (8): Enumerated values
//   - ClassVarLen (9): Variable-length data
//   - ClassArray (10): Fixed-size arrays
//
// # Layout Classes
//
// The [DataLayout] message describes one of three storage layouts:
//
//   - LayoutCompact (0): Data stored in the object header
//   - LayoutContiguous (1): Data in a single contiguous block
//   - LayoutChunked (2): Data in indexed chunks
//
// # Parsing
//
// Use [Parse] to parse a message from raw bytes:
//
//	msg, err := message.Parse(msgType, msgData, msgFlags, reader)
//
// The returned [Message] interface can be type-asserted to the specific
// message type based on its Type() method.
//
// # Writing
//
// Message serialization functions are available for writing HDF5 files:
//
//   - SerializeDataspace, SerializeDatatype, SerializeDataLayout, etc.
//   - These produce the raw bytes for embedding in object headers
package message
