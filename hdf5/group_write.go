package hdf5

import (
	"fmt"
	"path"

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

	return newGroup, nil
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
func (g *Group) loadExistingLinks() error {
	g.pendingLinks = make([]*message.Link, 0)

	// If we don't have a header loaded, try to load it
	if g.header == nil && g.file.reader != nil {
		header, err := object.Read(g.file.reader, g.addr)
		if err != nil {
			// If we can't read the header, start fresh (this is OK for new groups)
			return nil
		}
		g.header = header
	}

	// If we have a header, extract existing link messages
	if g.header != nil {
		linkMsgs := g.header.GetMessages(message.TypeLink)
		for _, msg := range linkMsgs {
			if linkMsg, ok := msg.(*message.Link); ok {
				g.pendingLinks = append(g.pendingLinks, linkMsg)
			}
		}
	}

	return nil
}

// rewriteHeader rewrites the group's object header with all pending links.
func (g *Group) rewriteHeader() error {
	// Create group header with LinkInfo and all links
	messages := object.NewGroupHeader(g.pendingLinks)

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

	// Update the link in parent's pending links
	for _, link := range parent.pendingLinks {
		if link.Name == name {
			link.ObjectAddress = newAddr
			break
		}
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

	// For now, if parent is root, return root
	if parentPath == "/" {
		return g.file.root
	}

	// For nested groups, we'd need to traverse
	// This is a simplification - proper implementation would maintain a group cache
	return nil
}
