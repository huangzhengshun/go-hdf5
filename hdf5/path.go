package hdf5

import (
	"fmt"
	"strings"
)

// ParseAttrPath parses an attribute path into object path and attribute name.
// Path format: /group/subgroup/object@attribute_name
//
// Examples:
//   - "/@root_attr" -> objectPath="/", attrName="root_attr"
//   - "/data@units" -> objectPath="/data", attrName="units"
//   - "/sensors/temp@calibration" -> objectPath="/sensors/temp", attrName="calibration"
//
// Returns an error if the path is invalid or missing the @ separator.
func ParseAttrPath(path string) (objectPath, attrName string, err error) {
	if path == "" {
		return "", "", fmt.Errorf("empty attribute path")
	}

	// Find the @ separator
	atIdx := strings.LastIndex(path, "@")
	if atIdx == -1 {
		return "", "", fmt.Errorf("attribute path must contain '@' separator: %s", path)
	}

	objectPath = path[:atIdx]
	attrName = path[atIdx+1:]

	if attrName == "" {
		return "", "", fmt.Errorf("attribute name cannot be empty: %s", path)
	}

	// Handle root case: "/@attr" -> objectPath should be "/"
	if objectPath == "" {
		objectPath = "/"
	}

	// Normalize object path
	if !strings.HasPrefix(objectPath, "/") {
		objectPath = "/" + objectPath
	}

	return objectPath, attrName, nil
}

// JoinAttrPath creates an attribute path from object path and attribute name.
func JoinAttrPath(objectPath, attrName string) string {
	if objectPath == "/" {
		return "/@" + attrName
	}
	return objectPath + "@" + attrName
}

// SplitPath splits a path into its components.
// Leading and trailing slashes are handled, empty components are removed.
//
// Examples:
//   - "/" -> []string{}
//   - "/foo" -> []string{"foo"}
//   - "/foo/bar" -> []string{"foo", "bar"}
func SplitPath(path string) []string {
	path = strings.Trim(path, "/")
	if path == "" {
		return []string{}
	}
	return strings.Split(path, "/")
}

// CleanPath normalizes a path, ensuring it starts with "/" and has no trailing slash.
func CleanPath(path string) string {
	if path == "" || path == "/" {
		return "/"
	}

	// Ensure leading slash
	if !strings.HasPrefix(path, "/") {
		path = "/" + path
	}

	// Remove trailing slash
	path = strings.TrimSuffix(path, "/")

	return path
}
