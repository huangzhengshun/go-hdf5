package hdf5

import (
	"path/filepath"
	"testing"
)

// TestReopenAndAdd tests the scenario of creating content, closing,
// reopening, adding more content, closing, and reopening to verify everything.
func TestReopenAndAdd(t *testing.T) {
	tmpDir := t.TempDir()
	filePath := filepath.Join(tmpDir, "reopen_test.h5")

	// Phase 1: Create initial structure
	f, err := Create(filePath)
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	grp1, err := f.Root().CreateGroup("group1")
	if err != nil {
		t.Fatalf("CreateGroup group1 failed: %v", err)
	}
	grp1.CreateAttribute("attr1", "first")
	grp1.CreateAttribute("count", int64(10))

	grp2, err := grp1.CreateGroup("subgroup1")
	if err != nil {
		t.Fatalf("CreateGroup subgroup1 failed: %v", err)
	}
	grp2.CreateAttribute("type", "initial")

	if err := f.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	// Phase 2: Reopen and add more content
	f2, err := OpenReadWrite(filePath)
	if err != nil {
		t.Fatalf("OpenReadWrite failed: %v", err)
	}

	// Add a new group to root
	grp3, err := f2.Root().CreateGroup("group2")
	if err != nil {
		t.Fatalf("CreateGroup group2 failed: %v", err)
	}
	grp3.CreateAttribute("name", "second group")

	// Add a new subgroup to existing group1
	grp1Reopened, err := f2.OpenGroup("/group1")
	if err != nil {
		t.Fatalf("OpenGroup /group1 failed: %v", err)
	}
	grp4, err := grp1Reopened.CreateGroup("subgroup2")
	if err != nil {
		t.Fatalf("CreateGroup subgroup2 failed: %v", err)
	}
	grp4.CreateAttribute("value", int64(99))

	// Add a new attribute to existing group1
	grp1Reopened.CreateAttribute("new_attr", "added on reopen")

	// Add a dataset to subgroup1
	subgrp1, err := f2.OpenGroup("/group1/subgroup1")
	if err != nil {
		t.Fatalf("OpenGroup /group1/subgroup1 failed: %v", err)
	}
	data := []int{1, 2, 3, 4, 5}
	_, err = subgrp1.CreateDataset("data", data)
	if err != nil {
		t.Fatalf("CreateDataset data failed: %v", err)
	}

	if err := f2.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	// Phase 3: Reopen and verify everything
	f3, err := Open(filePath)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f3.Close()

	// Verify group1 and its attributes
	g1, err := f3.OpenGroup("/group1")
	if err != nil {
		t.Fatalf("OpenGroup /group1 failed: %v", err)
	}
	if !g1.HasAttr("attr1") {
		t.Error("group1 should have attr1")
	}
	if !g1.HasAttr("count") {
		t.Error("group1 should have count")
	}
	if !g1.HasAttr("new_attr") {
		t.Error("group1 should have new_attr (added on reopen)")
	}

	// Verify subgroup1 and its dataset
	sg1, err := f3.OpenGroup("/group1/subgroup1")
	if err != nil {
		t.Fatalf("OpenGroup /group1/subgroup1 failed: %v", err)
	}
	if !sg1.HasAttr("type") {
		t.Error("subgroup1 should have type attr")
	}

	ds, err := f3.OpenDataset("/group1/subgroup1/data")
	if err != nil {
		t.Fatalf("OpenDataset /group1/subgroup1/data failed: %v", err)
	}
	_ = ds

	// Verify subgroup2
	sg2, err := f3.OpenGroup("/group1/subgroup2")
	if err != nil {
		t.Fatalf("OpenGroup /group1/subgroup2 failed: %v", err)
	}
	if !sg2.HasAttr("value") {
		t.Error("subgroup2 should have value attr")
	}

	// Verify group2
	g2, err := f3.OpenGroup("/group2")
	if err != nil {
		t.Fatalf("OpenGroup /group2 failed: %v", err)
	}
	if !g2.HasAttr("name") {
		t.Error("group2 should have name attr")
	}
}

// TestReopenAndAddDeep tests adding content at different depths on reopen
func TestReopenAndAddDeep(t *testing.T) {
	tmpDir := t.TempDir()
	filePath := filepath.Join(tmpDir, "deep_reopen_test.h5")

	// Phase 1: Create 3 levels
	f, err := Create(filePath)
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	// /a/b/c
	a, _ := f.Root().CreateGroup("a")
	a.CreateAttribute("level", int64(1))
	b, _ := a.CreateGroup("b")
	b.CreateAttribute("level", int64(2))
	c, _ := b.CreateGroup("c")
	c.CreateAttribute("level", int64(3))

	f.Close()

	// Phase 2: Reopen and add /a/b/d and /a/e
	f2, err := OpenReadWrite(filePath)
	if err != nil {
		t.Fatalf("OpenReadWrite failed: %v", err)
	}

	// Add /a/b/d
	b2, err := f2.OpenGroup("/a/b")
	if err != nil {
		t.Fatalf("OpenGroup /a/b failed: %v", err)
	}
	d, err := b2.CreateGroup("d")
	if err != nil {
		t.Fatalf("CreateGroup d failed: %v", err)
	}
	d.CreateAttribute("level", int64(3))

	// Add /a/e
	a2, err := f2.OpenGroup("/a")
	if err != nil {
		t.Fatalf("OpenGroup /a failed: %v", err)
	}
	e, err := a2.CreateGroup("e")
	if err != nil {
		t.Fatalf("CreateGroup e failed: %v", err)
	}
	e.CreateAttribute("level", int64(2))

	// Add dataset to /a/b/c
	c2, err := f2.OpenGroup("/a/b/c")
	if err != nil {
		t.Fatalf("OpenGroup /a/b/c failed: %v", err)
	}
	_, err = c2.CreateDataset("values", []float64{1.1, 2.2, 3.3})
	if err != nil {
		t.Fatalf("CreateDataset values failed: %v", err)
	}

	f2.Close()

	// Phase 3: Verify all
	f3, err := Open(filePath)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f3.Close()

	// Verify all groups exist
	paths := []string{"/a", "/a/b", "/a/b/c", "/a/b/d", "/a/e"}
	for _, p := range paths {
		_, err := f3.OpenGroup(p)
		if err != nil {
			t.Errorf("OpenGroup %s failed: %v", p, err)
		}
	}

	// Verify dataset
	_, err = f3.OpenDataset("/a/b/c/values")
	if err != nil {
		t.Errorf("OpenDataset /a/b/c/values failed: %v", err)
	}

	// Verify attributes
	g, _ := f3.OpenGroup("/a")
	if !g.HasAttr("level") {
		t.Error("/a should have level attr")
	}

	g, _ = f3.OpenGroup("/a/b/d")
	if !g.HasAttr("level") {
		t.Error("/a/b/d should have level attr")
	}
}

// TestMultipleFlushInOneSession tests calling Flush multiple times
func TestMultipleFlushInOneSession(t *testing.T) {
	tmpDir := t.TempDir()
	filePath := filepath.Join(tmpDir, "multi_flush_test.h5")

	f, err := Create(filePath)
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}
	defer f.Close()

	// Create group1 and flush
	grp1, _ := f.Root().CreateGroup("g1")
	grp1.CreateAttribute("a1", "v1")
	f.Flush()

	// Create group2 and flush
	grp2, _ := f.Root().CreateGroup("g2")
	grp2.CreateAttribute("a2", "v2")
	f.Flush()

	// Create subgroup and flush
	sub, _ := grp1.CreateGroup("sub")
	sub.CreateAttribute("a3", int64(3))
	f.Flush()

	// Verify by reopening
	f2, err := Open(filePath)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f2.Close()

	g1, err := f2.OpenGroup("/g1")
	if err != nil {
		t.Fatalf("OpenGroup /g1 failed: %v", err)
	}
	if !g1.HasAttr("a1") {
		t.Error("g1 should have a1")
	}

	_, err = f2.OpenGroup("/g1/sub")
	if err != nil {
		t.Fatalf("OpenGroup /g1/sub failed: %v", err)
	}

	g2, err := f2.OpenGroup("/g2")
	if err != nil {
		t.Fatalf("OpenGroup /g2 failed: %v", err)
	}
	if !g2.HasAttr("a2") {
		t.Error("g2 should have a2")
	}
}

// TestEmptyGroupAndAttrEdgeCases tests edge cases
func TestEmptyGroupAndAttrEdgeCases(t *testing.T) {
	tmpDir := t.TempDir()
	filePath := filepath.Join(tmpDir, "edge_test.h5")

	f, err := Create(filePath)
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}
	defer f.Close()

	// Create an empty group (no attributes, no subgroups)
	_, err = f.Root().CreateGroup("empty")
	if err != nil {
		t.Fatalf("CreateGroup empty failed: %v", err)
	}

	// Create a group with many attributes
	grp, err := f.Root().CreateGroup("many_attrs")
	if err != nil {
		t.Fatalf("CreateGroup many_attrs failed: %v", err)
	}
	for i := 0; i < 20; i++ {
		name := "attr_" + string(rune('a'+i))
		err := grp.CreateAttribute(name, "value_"+string(rune('a'+i)))
		if err != nil {
			t.Fatalf("CreateAttribute %s failed: %v", name, err)
		}
	}

	f.Flush()

	// Reopen and verify
	f2, err := Open(filePath)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f2.Close()

	_, err = f2.OpenGroup("/empty")
	if err != nil {
		t.Errorf("OpenGroup /empty failed: %v", err)
	}

	grp2, err := f2.OpenGroup("/many_attrs")
	if err != nil {
		t.Fatalf("OpenGroup /many_attrs failed: %v", err)
	}
	for i := 0; i < 20; i++ {
		name := "attr_" + string(rune('a'+i))
		if !grp2.HasAttr(name) {
			t.Errorf("many_attrs should have %s", name)
		}
	}
}

// TestReopenWithDatasetOnly tests reopening a file that has a dataset
// and adding a new group
func TestReopenWithDatasetOnly(t *testing.T) {
	tmpDir := t.TempDir()
	filePath := filepath.Join(tmpDir, "dataset_reopen_test.h5")

	// Phase 1: Create file with dataset
	f, err := Create(filePath)
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}
	grp, _ := f.Root().CreateGroup("data_group")
	_, err = grp.CreateDataset("ds1", []int{10, 20, 30})
	if err != nil {
		t.Fatalf("CreateDataset ds1 failed: %v", err)
	}
	f.Close()

	// Phase 2: Reopen and add a new group
	f2, err := OpenReadWrite(filePath)
	if err != nil {
		t.Fatalf("OpenReadWrite failed: %v", err)
	}
	_, err = f2.Root().CreateGroup("new_group")
	if err != nil {
		t.Fatalf("CreateGroup new_group failed: %v", err)
	}

	// Also add a dataset to the new group
	ng, err := f2.OpenGroup("/new_group")
	if err != nil {
		t.Fatalf("OpenGroup /new_group failed: %v", err)
	}
	_, err = ng.CreateDataset("ds2", []float64{1.5, 2.5})
	if err != nil {
		t.Fatalf("CreateDataset ds2 failed: %v", err)
	}
	f2.Close()

	// Phase 3: Verify
	f3, err := Open(filePath)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f3.Close()

	_, err = f3.OpenDataset("/data_group/ds1")
	if err != nil {
		t.Errorf("OpenDataset /data_group/ds1 failed: %v", err)
	}

	_, err = f3.OpenGroup("/new_group")
	if err != nil {
		t.Errorf("OpenGroup /new_group failed: %v", err)
	}

	_, err = f3.OpenDataset("/new_group/ds2")
	if err != nil {
		t.Errorf("OpenDataset /new_group/ds2 failed: %v", err)
	}
}
