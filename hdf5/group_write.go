package hdf5

import (
	"fmt"
	"path"
	"reflect"

	"github.com/huangzhengshun/go-hdf5/internal/dtype"
	"github.com/huangzhengshun/go-hdf5/internal/message"
	"github.com/huangzhengshun/go-hdf5/internal/object"
)

// pendingLink represents a link to be written to the parent group.
type pendingLink struct {
	link *message.Link
}

// CreateGroup creates a new subgroup with the given name.
func (g *Group) CreateGroup(name string) (*Group, error) {
	if !g.file.writable {
		return nil, fmt.Errorf("file is not writable")
	}

	if name == "" {
		return nil, fmt.Errorf("group name cannot be empty")
	}

	// Calculate the path for the new group
	newPath := path.Join(g.path, name)
	if g.path == "/" {
		newPath = "/" + name
	}

	// Create an empty group object header
	groupMessages := object.NewEmptyGroupHeader()

	// Calculate header size and allocate space
	headerSize := object.HeaderSize(g.file.writer, groupMessages)
	groupAddr := g.file.allocate(int64(headerSize))

	// Write the group object header
	w := g.file.writer.At(int64(groupAddr))
	if _, err := object.WriteHeader(w, groupMessages); err != nil {
		return nil, fmt.Errorf("writing group header: %w", err)
	}

	// Create a hard link from parent to this group
	link := message.NewHardLink(name, groupAddr)

	// Add the link to the parent group
	if err := g.addLink(link); err != nil {
		return nil, fmt.Errorf("adding link to parent: %w", err)
	}

	// Create the Group object
	newGroup := &Group{
		file:         g.file,
		path:         newPath,
		header:       nil, // Will be loaded on demand if needed
		addr:         groupAddr,
		pendingLinks: nil,
	}

	// Add to group cache for parent lookup
	g.file.mu.Lock()
	if g.file.groupCache == nil {
		g.file.groupCache = make(map[string]*Group)
	}
	g.file.groupCache[newPath] = newGroup
	g.file.mu.Unlock()

	return newGroup, nil
}

// CreateSoftLink creates a soft link pointing to the target path.
func (g *Group) CreateSoftLink(name, targetPath string) error {
	if !g.file.writable {
		return fmt.Errorf("file is not writable")
	}

	if name == "" {
		return fmt.Errorf("link name cannot be empty")
	}

	if targetPath == "" {
		return fmt.Errorf("target path cannot be empty")
	}

	link := message.NewSoftLink(name, targetPath)
	return g.addLink(link)
}

// CreateExternalLink creates an external link pointing to a dataset/group in another file.
func (g *Group) CreateExternalLink(name, externalFile, externalPath string) error {
	if !g.file.writable {
		return fmt.Errorf("file is not writable")
	}

	if name == "" {
		return fmt.Errorf("link name cannot be empty")
	}

	if externalFile == "" {
		return fmt.Errorf("external file path cannot be empty")
	}

	link := message.NewExternalLink(name, externalFile, externalPath)
	return g.addLink(link)
}

// CreateAttribute creates an attribute on the group.
func (g *Group) CreateAttribute(name string, value interface{}) error {
	if !g.file.writable {
		return fmt.Errorf("file is not writable")
	}

	if name == "" {
		return fmt.Errorf("attribute name cannot be empty")
	}

	val := reflect.ValueOf(value)
	if val.Kind() == reflect.Ptr {
		val = val.Elem()
	}

	var attr *message.Attribute
	var err error

	if val.Kind() == reflect.String {
		attr, err = createStringAttribute(name, val.String())
	} else if val.Kind() == reflect.Slice && val.Type().Elem().Kind() == reflect.String {
		attr, err = createStringArrayAttribute(name, val)
	} else {
		var dims []uint64
		var elemType reflect.Type

		switch val.Kind() {
		case reflect.Slice, reflect.Array:
			dims = []uint64{uint64(val.Len())}
			if val.Len() > 0 {
				elemType = val.Index(0).Type()
			} else {
				elemType = val.Type().Elem()
			}
		default:
			dims = nil
			elemType = val.Type()
		}

		datatype, err := dtype.GoTypeToDatatype(elemType)
		if err != nil {
			return fmt.Errorf("unsupported attribute type %v: %w", elemType, err)
		}

		var dataspace *message.Dataspace
		if dims == nil {
			dataspace = message.NewScalarDataspace()
		} else {
			dataspace = message.NewDataspace(dims, nil)
		}

		data, err := dtype.Encode(datatype, value)
		if err != nil {
			return fmt.Errorf("encoding attribute value: %w", err)
		}

		attr = message.NewAttribute(name, datatype, dataspace, data)
	}

	if err != nil {
		return err
	}

	g.pendingAttributes = append(g.pendingAttributes, attr)

	return g.rewriteHeader()
}

// addLink adds a link message to this group.
// For writable files, this updates the group's object header.
func (g *Group) addLink(link *message.Link) error {
	if !g.file.writable {
		return fmt.Errorf("file is not writable")
	}

	// If pendingLinks is nil, we need to load existing links from the header
	if g.pendingLinks == nil {
		if err := g.loadExistingLinks(); err != nil {
			return fmt.Errorf("loading existing links: %w", err)
		}
	}

	g.pendingLinks = append(g.pendingLinks, link)

	// Rewrite the group's object header with the new link
	return g.rewriteHeader()
}

// loadExistingLinks loads existing link messages from the group's object header.
// Supports both V2 Link messages and V1 symbol table entries.
func (g *Group) loadExistingLinks() error {
	g.pendingLinks = make([]*message.Link, 0)

	// If we don't have a header loaded, try to load it
	if g.header == nil && g.file.reader != nil {
		header, err := object.Read(g.file.reader, g.addr)
		if err != nil {
			return nil
		}
		g.header = header
	}

	if g.header == nil {
		return nil
	}

	// Load V2 Link messages
	linkMsgs := g.header.GetMessages(message.TypeLink)
	for _, msg := range linkMsgs {
		if linkMsg, ok := msg.(*message.Link); ok {
			g.pendingLinks = append(g.pendingLinks, linkMsg)
		}
	}

	// Load V1 symbol table entries if no V2 links found
	if len(g.pendingLinks) == 0 {
		symMsg := g.header.GetMessage(message.TypeSymbolTable)
		if symMsg != nil {
			symTable := symMsg.(*message.SymbolTable)
			entries, err := g.getMembersV1(symTable)
			if err == nil {
				for _, entry := range entries {
					var link *message.Link
					if entry.LinkType == 1 {
						link = message.NewSoftLink(entry.Name, entry.SoftLinkValue)
					} else {
						link = message.NewHardLink(entry.Name, entry.ObjectAddress)
					}
					g.pendingLinks = append(g.pendingLinks, link)
				}
			}
		} else if g.path == "/" && g.file.superblock.RootGroupBTreeAddress != 0 {
			symTable := &message.SymbolTable{
				BTreeAddress:     g.file.superblock.RootGroupBTreeAddress,
				LocalHeapAddress: g.file.superblock.RootGroupLocalHeapAddress,
			}
			entries, err := g.getMembersV1(symTable)
			if err == nil {
				for _, entry := range entries {
					var link *message.Link
					if entry.LinkType == 1 {
						link = message.NewSoftLink(entry.Name, entry.SoftLinkValue)
					} else {
						link = message.NewHardLink(entry.Name, entry.ObjectAddress)
					}
					g.pendingLinks = append(g.pendingLinks, link)
				}
			}
		}
	}

	return nil
}

// rewriteHeader rewrites the group's object header with all pending links and attributes.
func (g *Group) rewriteHeader() error {
	// Create group header with LinkInfo and all links
	messages := object.NewGroupHeader(g.pendingLinks)

	// Add pending attributes
	for _, attr := range g.pendingAttributes {
		messages = append(messages, attr)
	}

	// Calculate new header size with minimum chunk size for h5py compatibility
	headerSize := object.HeaderSizeWithMinChunk(g.file.writer, messages, object.MinGroupChunkSize)

	// Allocate new space (we can't resize in place, so allocate new)
	newAddr := g.file.allocate(int64(headerSize))

	// Write the new header
	w := g.file.writer.At(int64(newAddr))
	if _, err := object.WriteHeaderWithMinChunk(w, messages, object.MinGroupChunkSize); err != nil {
		return err
	}

	// Update our address
	oldAddr := g.addr
	g.addr = newAddr

	// If this is the root group, update the superblock
	if g.path == "/" {
		g.file.superblock.RootGroupAddress = newAddr
	} else {
		// Update parent's link to point to new address
		if err := g.updateParentLink(oldAddr, newAddr); err != nil {
			return err
		}
	}

	return nil
}

// updateParentLink updates the parent group's link to point to the new address.
func (g *Group) updateParentLink(oldAddr, newAddr uint64) error {
	// Find parent group
	parentPath := path.Dir(g.path)
	if parentPath == "" || parentPath == "." {
		parentPath = "/"
	}

	// Get the name of this group
	name := path.Base(g.path)

	// Find parent in our hierarchy
	parent := g.findParent()
	if parent == nil {
		return nil // Root group, no parent
	}

	// Ensure parent's existing links are loaded
	if parent.pendingLinks == nil {
		if err := parent.loadExistingLinks(); err != nil {
			return fmt.Errorf("loading parent links: %w", err)
		}
	}

	// Update the link in parent's pending links
	found := false
	for _, link := range parent.pendingLinks {
		if link.Name == name {
			link.ObjectAddress = newAddr
			found = true
			break
		}
	}

	// If not found in pendingLinks, create a new link
	if !found {
		link := message.NewHardLink(name, newAddr)
		parent.pendingLinks = append(parent.pendingLinks, link)
	}

	// Rewrite parent's header
	return parent.rewriteHeader()
}

// findParent finds the parent group in the file's group hierarchy.
func (g *Group) findParent() *Group {
	if g.path == "/" {
		return nil
	}

	parentPath := path.Dir(g.path)
	if parentPath == "" || parentPath == "." {
		parentPath = "/"
	}

	// Try to find parent in group cache
	g.file.mu.RLock()
	if g.file.groupCache != nil {
		if parent, ok := g.file.groupCache[parentPath]; ok {
			g.file.mu.RUnlock()
			return parent
		}
	}
	g.file.mu.RUnlock()

	// Fallback: if parent is root, return root
	if parentPath == "/" {
		return g.file.root
	}

	// If not in cache, try to open the parent group from file
	parent, err := g.file.OpenGroup(parentPath)
	if err != nil {
		return nil
	}

	// Add to cache for future lookups
	g.file.mu.Lock()
	if g.file.groupCache == nil {
		g.file.groupCache = make(map[string]*Group)
	}
	g.file.groupCache[parentPath] = parent
	g.file.mu.Unlock()

	return parent
}
