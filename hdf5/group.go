package hdf5

import (
	"fmt"
	"path"

	"github.com/huangzhengshun/go-hdf5/internal/btree"
	"github.com/huangzhengshun/go-hdf5/internal/heap"
	"github.com/huangzhengshun/go-hdf5/internal/message"
	"github.com/huangzhengshun/go-hdf5/internal/object"
)

// Group represents an HDF5 group.
type Group struct {
	file   *File
	path   string
	header *object.Header
	addr   uint64 // Object header address (for write support)

	// Write support fields
	pendingLinks     []*message.Link        // Links to be written
	pendingAttributes []*message.Attribute  // Attributes to be written
}

// ObjectType indicates the type of an HDF5 object.
type ObjectType string

const (
	ObjectTypeGroup   ObjectType = "group"
	ObjectTypeDataset ObjectType = "dataset"
	ObjectTypeUnknown ObjectType = "unknown"
)

// MemberInfo contains information about a group member.
type MemberInfo struct {
	Name     string     // Name of the member
	Type     ObjectType // Type of the object ("group", "dataset", or "unknown")
	LinkType string     // Type of the link ("hard", "soft", or "external")
}

// linkResolution holds the result of resolving a link.
type linkResolution struct {
	address   uint64 // Object address
	isDataset bool   // True if target is a dataset
	file      *File  // Target file (nil = same file, non-nil = external file)
}

// Name returns the group name (last component of path).
func (g *Group) Name() string {
	if g.path == "/" {
		return "/"
	}
	return path.Base(g.path)
}

// Path returns the full path to this group.
func (g *Group) Path() string {
	return g.path
}

// OpenGroup opens a subgroup by relative path.
func (g *Group) OpenGroup(relativePath string) (*Group, error) {
	obj, err := g.open(relativePath)
	if err != nil {
		return nil, err
	}

	group, ok := obj.(*Group)
	if !ok {
		return nil, ErrNotGroup
	}
	return group, nil
}

// OpenDataset opens a dataset by relative path.
func (g *Group) OpenDataset(relativePath string) (*Dataset, error) {
	obj, err := g.open(relativePath)
	if err != nil {
		return nil, err
	}

	dataset, ok := obj.(*Dataset)
	if !ok {
		return nil, ErrNotDataset
	}
	return dataset, nil
}

// open opens an object by relative path.
func (g *Group) open(relativePath string) (interface{}, error) {
	parts := splitPath(relativePath)
	if len(parts) == 0 {
		return g, nil
	}

	current := g
	visited := make(map[string]bool)

	for i, name := range parts {
		res, err := current.findChildFull(name, visited)
		if err != nil {
			return nil, fmt.Errorf("finding %q: %w", name, err)
		}

		// Determine which file to use for opening the object
		targetFile := current.file
		if res.file != nil {
			targetFile = res.file
		}

		fullPath := path.Join(current.path, name)

		// If this is the last component, open as appropriate type
		if i == len(parts)-1 {
			if res.isDataset {
				return targetFile.openDatasetAt(res.address, fullPath)
			}
			return targetFile.openGroupAt(res.address, fullPath)
		}

		// Otherwise, must be a group to continue traversal
		if res.isDataset {
			return nil, fmt.Errorf("%q is not a group", fullPath)
		}

		nextGroup, err := targetFile.openGroupAt(res.address, fullPath)
		if err != nil {
			return nil, err
		}
		current = nextGroup
	}

	return current, nil
}

// findChild finds a child object by name and returns its address.
// Returns (address, isDataset, error).
func (g *Group) findChild(name string) (uint64, bool, error) {
	res, err := g.findChildFull(name, make(map[string]bool))
	if err != nil {
		return 0, false, err
	}
	return res.address, res.isDataset, nil
}

// findChildFull finds a child and returns full resolution info including external file.
func (g *Group) findChildFull(name string, visited map[string]bool) (*linkResolution, error) {
	// Try to find via Link messages (v2 groups)
	for _, msg := range g.header.GetMessages(message.TypeLink) {
		link := msg.(*message.Link)
		if link.Name == name {
			return g.resolveLink(link, visited)
		}
	}

	// Try symbol table (v1 groups) - requires B-tree traversal
	symMsg := g.header.GetMessage(message.TypeSymbolTable)
	if symMsg != nil {
		symTable := symMsg.(*message.SymbolTable)
		return g.findChildV1Full(name, symTable, visited)
	}

	// Fallback for root group: use cached addresses from superblock scratch pad
	if g.path == "/" && g.file.superblock.RootGroupBTreeAddress != 0 {
		symTable := &message.SymbolTable{
			BTreeAddress:     g.file.superblock.RootGroupBTreeAddress,
			LocalHeapAddress: g.file.superblock.RootGroupLocalHeapAddress,
		}
		return g.findChildV1Full(name, symTable, visited)
	}

	return nil, ErrNotFound
}

// resolveLink resolves a link to get the target object's address.
func (g *Group) resolveLink(link *message.Link, visited map[string]bool) (*linkResolution, error) {
	switch {
	case link.IsHard():
		isDataset, err := g.isDataset(link.ObjectAddress)
		if err != nil {
			return nil, err
		}
		return &linkResolution{
			address:   link.ObjectAddress,
			isDataset: isDataset,
			file:      nil, // Same file
		}, nil

	case link.IsSoft():
		targetPath := link.SoftLinkValue
		if len(visited) >= MaxLinkDepth {
			return nil, ErrLinkDepth
		}
		if visited[targetPath] {
			return nil, fmt.Errorf("circular soft link detected: %s", targetPath)
		}
		visited[targetPath] = true
		res, err := g.file.findByAbsolutePathFull(targetPath, visited)
		if err != nil {
			return nil, err
		}
		return res, nil

	case link.IsExternal():
		addr, isDs, extFile, err := g.file.resolveExternalLink(
			link.ExternalFile, link.ExternalPath, visited)
		if err != nil {
			return nil, err
		}
		return &linkResolution{
			address:   addr,
			isDataset: isDs,
			file:      extFile,
		}, nil

	default:
		return nil, fmt.Errorf("unknown link type: %d", link.LinkType)
	}
}

// findChildV1 finds a child in a v1 group using the symbol table.
func (g *Group) findChildV1(name string, symTable *message.SymbolTable) (uint64, bool, error) {
	res, err := g.findChildV1Full(name, symTable, make(map[string]bool))
	if err != nil {
		return 0, false, err
	}
	return res.address, res.isDataset, nil
}

// findChildV1Full finds a child in a v1 group with full resolution info.
func (g *Group) findChildV1Full(name string, symTable *message.SymbolTable, visited map[string]bool) (*linkResolution, error) {
	// Read the local heap to get string names
	localHeap, err := heap.ReadLocalHeap(g.file.reader, symTable.LocalHeapAddress)
	if err != nil {
		return nil, fmt.Errorf("reading local heap: %w", err)
	}

	// Read the B-tree to get group entries
	entries, err := btree.ReadGroupEntries(g.file.reader, symTable.BTreeAddress, localHeap)
	if err != nil {
		return nil, fmt.Errorf("reading B-tree: %w", err)
	}

	// Find the named entry
	for _, entry := range entries {
		if entry.Name == name {
			// Check if this is a soft link
			if entry.LinkType == 1 {
				// Soft link - resolve the target path
				targetPath := entry.SoftLinkValue
				if len(visited) >= MaxLinkDepth {
					return nil, ErrLinkDepth
				}
				if visited[targetPath] {
					return nil, fmt.Errorf("circular soft link detected: %s", targetPath)
				}
				visited[targetPath] = true
				addr, isDs, err := g.file.findByAbsolutePath(targetPath, visited)
				if err != nil {
					return nil, err
				}
				return &linkResolution{
					address:   addr,
					isDataset: isDs,
					file:      nil, // Same file (v1 groups don't support external links)
				}, nil
			}

			// Hard link - return object address
			isDataset, err := g.isDataset(entry.ObjectAddress)
			if err != nil {
				return nil, err
			}
			return &linkResolution{
				address:   entry.ObjectAddress,
				isDataset: isDataset,
				file:      nil,
			}, nil
		}
	}

	return nil, ErrNotFound
}

// isDataset checks if an object at the given address is a dataset.
func (g *Group) isDataset(address uint64) (bool, error) {
	header, err := object.Read(g.file.reader, address)
	if err != nil {
		return false, err
	}

	// A dataset has a dataspace message
	return header.GetMessage(message.TypeDataspace) != nil, nil
}

// Members returns the names of all members (groups and datasets) in this group.
func (g *Group) Members() ([]string, error) {
	var names []string

	// Collect from Link messages (v2 groups)
	for _, msg := range g.header.GetMessages(message.TypeLink) {
		link := msg.(*message.Link)
		names = append(names, link.Name)
	}

	// If using symbol table (v1 groups), traverse the B-tree
	if len(names) == 0 {
		symMsg := g.header.GetMessage(message.TypeSymbolTable)
		if symMsg != nil {
			symTable := symMsg.(*message.SymbolTable)
			entries, err := g.getMembersV1(symTable)
			if err != nil {
				return nil, err
			}
			for _, entry := range entries {
				names = append(names, entry.Name)
			}
		} else if g.path == "/" && g.file.superblock.RootGroupBTreeAddress != 0 {
			// Fallback for root group: use cached addresses from superblock scratch pad
			symTable := &message.SymbolTable{
				BTreeAddress:     g.file.superblock.RootGroupBTreeAddress,
				LocalHeapAddress: g.file.superblock.RootGroupLocalHeapAddress,
			}
			entries, err := g.getMembersV1(symTable)
			if err != nil {
				return nil, err
			}
			for _, entry := range entries {
				names = append(names, entry.Name)
			}
		}
	}

	return names, nil
}

// getMembersV1 gets all members from a v1 group using the symbol table.
func (g *Group) getMembersV1(symTable *message.SymbolTable) ([]btree.GroupEntry, error) {
	// Read the local heap to get string names
	localHeap, err := heap.ReadLocalHeap(g.file.reader, symTable.LocalHeapAddress)
	if err != nil {
		return nil, fmt.Errorf("reading local heap: %w", err)
	}

	// Read the B-tree to get group entries
	return btree.ReadGroupEntries(g.file.reader, symTable.BTreeAddress, localHeap)
}

// MembersInfo returns detailed information about all members in this group.
// This includes the object type and link type for each member.
func (g *Group) MembersInfo() ([]MemberInfo, error) {
	var members []MemberInfo

	// Collect from Link messages (v2 groups)
	for _, msg := range g.header.GetMessages(message.TypeLink) {
		link := msg.(*message.Link)
		info := MemberInfo{
			Name:     link.Name,
			LinkType: g.linkTypeString(link),
		}

		// Determine object type by checking if it's a dataset or group
		objType := ObjectTypeUnknown
		if link.IsHard() {
			isDs, err := g.isDataset(link.ObjectAddress)
			if err == nil {
				if isDs {
					objType = ObjectTypeDataset
				} else {
					objType = ObjectTypeGroup
				}
			}
		} else if link.IsSoft() || link.IsExternal() {
			// For soft/external links, try to resolve and check type
			res, err := g.resolveLink(link, make(map[string]bool))
			if err == nil {
				if res.isDataset {
					objType = ObjectTypeDataset
				} else {
					objType = ObjectTypeGroup
				}
			}
		}
		info.Type = objType
		members = append(members, info)
	}

	// If using symbol table (v1 groups), traverse the B-tree
	if len(members) == 0 {
		symMsg := g.header.GetMessage(message.TypeSymbolTable)
		var symTable *message.SymbolTable
		if symMsg != nil {
			symTable = symMsg.(*message.SymbolTable)
		} else if g.path == "/" && g.file.superblock.RootGroupBTreeAddress != 0 {
			// Fallback for root group
			symTable = &message.SymbolTable{
				BTreeAddress:     g.file.superblock.RootGroupBTreeAddress,
				LocalHeapAddress: g.file.superblock.RootGroupLocalHeapAddress,
			}
		}

		if symTable != nil {
			entries, err := g.getMembersV1(symTable)
			if err != nil {
				return nil, err
			}
			for _, entry := range entries {
				info := MemberInfo{
					Name: entry.Name,
				}

				// Determine link type
				if entry.LinkType == 1 {
					info.LinkType = "soft"
				} else {
					info.LinkType = "hard"
				}

				// Determine object type
				objType := ObjectTypeUnknown
				if entry.LinkType == 0 && entry.ObjectAddress != 0 {
					isDs, err := g.isDataset(entry.ObjectAddress)
					if err == nil {
						if isDs {
							objType = ObjectTypeDataset
						} else {
							objType = ObjectTypeGroup
						}
					}
				}
				info.Type = objType
				members = append(members, info)
			}
		}
	}

	return members, nil
}

// linkTypeString returns a string representation of the link type.
func (g *Group) linkTypeString(link *message.Link) string {
	switch {
	case link.IsHard():
		return "hard"
	case link.IsSoft():
		return "soft"
	case link.IsExternal():
		return "external"
	default:
		return "unknown"
	}
}

// NumObjects returns the number of objects in this group.
func (g *Group) NumObjects() (int, error) {
	members, err := g.Members()
	if err != nil {
		return 0, err
	}
	return len(members), nil
}

// Attrs returns the attribute names for this group.
func (g *Group) Attrs() []string {
	var names []string
	for _, msg := range g.header.GetMessages(message.TypeAttribute) {
		attr := msg.(*message.Attribute)
		names = append(names, attr.Name)
	}
	return names
}

// Attr returns an attribute by name, or nil if not found.
func (g *Group) Attr(name string) *Attribute {
	for _, msg := range g.header.GetMessages(message.TypeAttribute) {
		attr := msg.(*message.Attribute)
		if attr.Name == name {
			return &Attribute{msg: attr, reader: g.file.reader}
		}
	}
	return nil
}

// HasAttr returns true if the group has an attribute with the given name.
func (g *Group) HasAttr(name string) bool {
	return g.Attr(name) != nil
}
