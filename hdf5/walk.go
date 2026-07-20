package hdf5

import (
	"path"
)

// WalkFunc is called for each object during traversal.
// path is the full path to the object.
// obj is either *Group or *Dataset.
// err is any error encountered opening the object.
// Return nil to continue walking, or an error to stop.
type WalkFunc func(path string, obj interface{}, err error) error

// Walk traverses all objects (groups and datasets) in the hierarchy starting from g.
// The callback is called for each group and dataset, including the starting group.
//
// Example:
//
//	Walk(root, func(path string, obj interface{}, err error) error {
//	    if err != nil {
//	        return err // or skip: return nil
//	    }
//	    switch o := obj.(type) {
//	    case *Group:
//	        fmt.Println("Group:", path)
//	    case *Dataset:
//	        fmt.Println("Dataset:", path, "shape:", o.Shape())
//	    }
//	    return nil
//	})
func Walk(g *Group, fn WalkFunc) error {
	return walkGroup(g, fn)
}

// walkGroup recursively walks a group and its children.
func walkGroup(g *Group, fn WalkFunc) error {
	// Call fn for this group first
	if err := fn(g.Path(), g, nil); err != nil {
		return err
	}

	// Get all members
	members, err := g.Members()
	if err != nil {
		return err
	}

	// Process each child
	for _, name := range members {
		childPath := path.Join(g.Path(), name)

		// Try as group first
		childGroup, err := g.OpenGroup(name)
		if err == nil {
			// It's a group - recurse
			if err := walkGroup(childGroup, fn); err != nil {
				return err
			}
			continue
		}

		// Try as dataset
		dataset, err := g.OpenDataset(name)
		if err == nil {
			if err := fn(childPath, dataset, nil); err != nil {
				return err
			}
			continue
		}

		// Could not open as either - call fn with error
		if err := fn(childPath, nil, err); err != nil {
			return err
		}
	}

	return nil
}

// AttrInfo contains information about an attribute during walking.
type AttrInfo struct {
	// Path is the full attribute path (e.g., "/group/dataset@attr")
	Path string

	// ObjectPath is the path to the object containing this attribute
	ObjectPath string

	// ObjectType is "group" or "dataset"
	ObjectType string

	// Name is the attribute name
	Name string

	// Attr provides access to the full attribute for detailed reading
	Attr *Attribute

	// Value contains the auto-read attribute value (nil on read error)
	Value interface{}

	// Err contains any error from reading the attribute value
	Err error
}

// WalkAttrsFunc is the callback function type for WalkAttrs.
// Return nil to continue walking, or an error to stop.
type WalkAttrsFunc func(info AttrInfo) error

// WalkAttrs recursively walks all attributes in the file.
// The callback is called for each attribute on groups and datasets.
//
// Example:
//
//	f.WalkAttrs(func(info hdf5.AttrInfo) error {
//	    fmt.Printf("%s = %v\n", info.Path, info.Value)
//	    return nil
//	})
func (f *File) WalkAttrs(fn WalkAttrsFunc) error {
	if f.closed {
		return ErrClosed
	}
	return f.walkGroupAttrs(f.root, fn)
}

// walkGroupAttrs recursively walks attributes in a group and its children.
func (f *File) walkGroupAttrs(g *Group, fn WalkAttrsFunc) error {
	// Process attributes on this group
	for _, name := range g.Attrs() {
		attr := g.Attr(name)
		info := AttrInfo{
			Path:       JoinAttrPath(g.Path(), name),
			ObjectPath: g.Path(),
			ObjectType: "group",
			Name:       name,
			Attr:       attr,
		}

		// Try to read the value
		if attr != nil {
			val, err := attr.Value()
			info.Value = val
			info.Err = err
		}

		if err := fn(info); err != nil {
			return err
		}
	}

	// Get all members of this group
	members, err := g.Members()
	if err != nil {
		return err
	}

	// Process each child
	for _, name := range members {
		childPath := path.Join(g.Path(), name)

		// Try as group first
		childGroup, err := g.OpenGroup(name)
		if err == nil {
			// It's a group - recurse
			if err := f.walkGroupAttrs(childGroup, fn); err != nil {
				return err
			}
			continue
		}

		// Try as dataset
		dataset, err := g.OpenDataset(name)
		if err != nil {
			// Skip objects we can't open
			continue
		}

		// Process attributes on this dataset
		for _, attrName := range dataset.Attrs() {
			attr := dataset.Attr(attrName)
			info := AttrInfo{
				Path:       JoinAttrPath(childPath, attrName),
				ObjectPath: childPath,
				ObjectType: "dataset",
				Name:       attrName,
				Attr:       attr,
			}

			// Try to read the value
			if attr != nil {
				val, err := attr.Value()
				info.Value = val
				info.Err = err
			}

			if err := fn(info); err != nil {
				return err
			}
		}
	}

	return nil
}

// ErrStopWalk can be returned from WalkAttrsFunc to stop walking without an error.
var ErrStopWalk = &walkStopError{}

type walkStopError struct{}

func (e *walkStopError) Error() string { return "walk stopped" }

// IsStopWalk returns true if the error is ErrStopWalk.
func IsStopWalk(err error) bool {
	_, ok := err.(*walkStopError)
	return ok
}
