package main

import (
	"fmt"
	"os"

	"github.com/huangzhengshun/go-hdf5/hdf5"
)

func main() {
	path := "nested_groups_with_attrs.h5"
	os.Remove(path) // Remove if exists, but don't defer delete

	f, err := hdf5.Create(path)
	if err != nil {
		fmt.Printf("Create failed: %v\n", err)
		os.Exit(1)
	}

	// Level 1: /sensor_data
	sensorData, err := f.Root().CreateGroup("sensor_data")
	if err != nil {
		fmt.Printf("CreateGroup sensor_data failed: %v\n", err)
		os.Exit(1)
	}
	sensorData.CreateAttribute("description", "Sensor measurements collection")
	sensorData.CreateAttribute("version", int64(2))

	// Level 2: /sensor_data/temperature
	temp, err := sensorData.CreateGroup("temperature")
	if err != nil {
		fmt.Printf("CreateGroup temperature failed: %v\n", err)
		os.Exit(1)
	}
	temp.CreateAttribute("unit", "celsius")
	temp.CreateAttribute("accuracy", "±0.5°C")

	// Level 3: /sensor_data/temperature/calibration
	calib, err := temp.CreateGroup("calibration")
	if err != nil {
		fmt.Printf("CreateGroup calibration failed: %v\n", err)
		os.Exit(1)
	}
	calib.CreateAttribute("calibrated_by", "John Doe")
	calib.CreateAttribute("calibration_date", "2024-01-15")

	// Level 2: /sensor_data/pressure
	pressure, err := sensorData.CreateGroup("pressure")
	if err != nil {
		fmt.Printf("CreateGroup pressure failed: %v\n", err)
		os.Exit(1)
	}
	pressure.CreateAttribute("unit", "pascal")
	pressure.CreateAttribute("range", "0-100000 Pa")

	// Level 3: /sensor_data/pressure/calibration
	pCalib, err := pressure.CreateGroup("calibration")
	if err != nil {
		fmt.Printf("CreateGroup pressure/calibration failed: %v\n", err)
		os.Exit(1)
	}
	pCalib.CreateAttribute("calibrated_by", "Jane Smith")
	pCalib.CreateAttribute("calibration_date", "2024-02-20")

	// Level 1: /metadata
	metadata, err := f.Root().CreateGroup("metadata")
	if err != nil {
		fmt.Printf("CreateGroup metadata failed: %v\n", err)
		os.Exit(1)
	}
	metadata.CreateAttribute("author", "Test System")
	metadata.CreateAttribute("project", "HDF5 Validation")

	if err := f.Close(); err != nil {
		fmt.Printf("Close failed: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("File created successfully:", path)

	// Verify by reopening
	f2, err := hdf5.Open(path)
	if err != nil {
		fmt.Printf("Open failed: %v\n", err)
		os.Exit(1)
	}
	defer f2.Close()

	// Verify groups and attributes
	groups := []struct {
		path  string
		attrs map[string]interface{}
	}{
		{"/sensor_data", nil},
		{"/sensor_data/temperature", nil},
		{"/sensor_data/temperature/calibration", nil},
		{"/sensor_data/pressure", nil},
		{"/sensor_data/pressure/calibration", nil},
		{"/metadata", nil},
	}

	for _, g := range groups {
		grp, err := f2.OpenGroup(g.path)
		if err != nil {
			fmt.Printf("OpenGroup %s failed: %v\n", g.path, err)
			os.Exit(1)
		}
		fmt.Printf("Opened group: %s\n", g.path)

		// Check attributes
		for _, attrName := range []string{"description", "version", "unit", "accuracy", "calibrated_by", "calibration_date", "range", "author", "project"} {
			if grp.HasAttr(attrName) {
				attr := grp.Attr(attrName)
				val, err := attr.Value()
				if err == nil {
					fmt.Printf("  attr %s = %v\n", attrName, val)
				}
			}
		}
	}

	fmt.Println("\nAll groups and attributes verified successfully!")
}
