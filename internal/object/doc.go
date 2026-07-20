// Package object handles parsing of HDF5 object headers.
//
// Every HDF5 object (group, dataset, committed datatype) has an object header
// that contains metadata in the form of header messages. This package provides
// functionality to read object headers and access their messages.
//
// # Object Header Versions
//
// HDF5 defines two object header versions:
//
//   - Version 1: Used in older files (superblock v0/v1). Messages are stored
//     in a linked list of header continuation blocks with 8-byte alignment.
//     The first byte of the header is the version number (1).
//
//   - Version 2 (signature "OHDR"): Used in newer files (superblock v2/v3).
//     Provides better space efficiency with variable-size message headers
//     and optional checksums. Supports timestamps and attribute storage hints.
//
// The [Read] function automatically detects the header version and parses
// accordingly.
//
// # Header Structure
//
// An object header contains:
//
//   - Version and flags (v2: timestamps, attribute hints)
//   - Reference count for the object
//   - Sequence of header messages (dataspace, datatype, layout, etc.)
//   - Optional continuation blocks for overflow messages
//
// # Usage
//
// Read an object header at a known address:
//
//	header, err := object.Read(reader, objectAddress)
//
// Access specific messages:
//
//	dataspace := header.Dataspace()
//	datatype := header.Datatype()
//	layout := header.DataLayout()
//	filterPipeline := header.FilterPipeline()
//
// Or use generic message access:
//
//	msg := header.GetMessage(message.TypeDataspace)
//	allAttrs := header.GetMessages(message.TypeAttribute)
//
// # Key Types
//
//   - [Header]: Parsed object header with version, flags, and messages
//   - [Read]: Parses an object header at a given file address
//
// # Errors
//
//   - [ErrInvalidHeader]: Header format not recognized
//   - [ErrUnsupportedVersion]: Header version not supported
//   - [ErrChecksumMismatch]: V2 header checksum verification failed
package object
