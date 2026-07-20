package dtype

import (
	"reflect"
	"testing"

	"github.com/huangzhengshun/go-hdf5/internal/message"
)

func TestGoTypeFixedPoint(t *testing.T) {
	tests := []struct {
		name     string
		dt       *message.Datatype
		expected reflect.Type
	}{
		{"int8", &message.Datatype{Class: message.ClassFixedPoint, Size: 1, Signed: true}, reflect.TypeOf(int8(0))},
		{"uint8", &message.Datatype{Class: message.ClassFixedPoint, Size: 1, Signed: false}, reflect.TypeOf(uint8(0))},
		{"int16", &message.Datatype{Class: message.ClassFixedPoint, Size: 2, Signed: true}, reflect.TypeOf(int16(0))},
		{"uint16", &message.Datatype{Class: message.ClassFixedPoint, Size: 2, Signed: false}, reflect.TypeOf(uint16(0))},
		{"int32", &message.Datatype{Class: message.ClassFixedPoint, Size: 4, Signed: true}, reflect.TypeOf(int32(0))},
		{"uint32", &message.Datatype{Class: message.ClassFixedPoint, Size: 4, Signed: false}, reflect.TypeOf(uint32(0))},
		{"int64", &message.Datatype{Class: message.ClassFixedPoint, Size: 8, Signed: true}, reflect.TypeOf(int64(0))},
		{"uint64", &message.Datatype{Class: message.ClassFixedPoint, Size: 8, Signed: false}, reflect.TypeOf(uint64(0))},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := GoType(tt.dt)
			if err != nil {
				t.Fatalf("GoType failed: %v", err)
			}
			if got != tt.expected {
				t.Errorf("expected %v, got %v", tt.expected, got)
			}
		})
	}
}

func TestGoTypeFloatPoint(t *testing.T) {
	tests := []struct {
		name     string
		dt       *message.Datatype
		expected reflect.Type
	}{
		{"float32", &message.Datatype{Class: message.ClassFloatPoint, Size: 4}, reflect.TypeOf(float32(0))},
		{"float64", &message.Datatype{Class: message.ClassFloatPoint, Size: 8}, reflect.TypeOf(float64(0))},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := GoType(tt.dt)
			if err != nil {
				t.Fatalf("GoType failed: %v", err)
			}
			if got != tt.expected {
				t.Errorf("expected %v, got %v", tt.expected, got)
			}
		})
	}
}

func TestGoTypeString(t *testing.T) {
	dt := &message.Datatype{Class: message.ClassString, Size: 10}
	got, err := GoType(dt)
	if err != nil {
		t.Fatalf("GoType failed: %v", err)
	}
	if got != reflect.TypeOf("") {
		t.Errorf("expected string type, got %v", got)
	}
}

func TestGoTypeVarLenString(t *testing.T) {
	dt := &message.Datatype{Class: message.ClassVarLen, IsVarLenString: true}
	got, err := GoType(dt)
	if err != nil {
		t.Fatalf("GoType failed: %v", err)
	}
	if got != reflect.TypeOf("") {
		t.Errorf("expected string type, got %v", got)
	}
}

func TestGoTypeArray(t *testing.T) {
	// Array of 5 int32s
	dt := &message.Datatype{
		Class:     message.ClassArray,
		ArrayDims: []uint32{5},
		BaseType: &message.Datatype{
			Class:  message.ClassFixedPoint,
			Size:   4,
			Signed: true,
		},
	}

	got, err := GoType(dt)
	if err != nil {
		t.Fatalf("GoType failed: %v", err)
	}

	expected := reflect.TypeOf([5]int32{})
	if got != expected {
		t.Errorf("expected %v, got %v", expected, got)
	}
}

func TestConvertInt32(t *testing.T) {
	dt := &message.Datatype{
		Class:     message.ClassFixedPoint,
		Size:      4,
		Signed:    true,
		ByteOrder: message.OrderLE,
	}

	// 3 int32 values: 1, 2, 3
	data := []byte{
		0x01, 0x00, 0x00, 0x00,
		0x02, 0x00, 0x00, 0x00,
		0x03, 0x00, 0x00, 0x00,
	}

	var result []int32
	err := Convert(dt, data, 3, &result)
	if err != nil {
		t.Fatalf("Convert failed: %v", err)
	}

	if len(result) != 3 {
		t.Fatalf("expected 3 elements, got %d", len(result))
	}

	expected := []int32{1, 2, 3}
	for i, v := range expected {
		if result[i] != v {
			t.Errorf("result[%d] = %d, want %d", i, result[i], v)
		}
	}
}

func TestConvertFloat64(t *testing.T) {
	dt := &message.Datatype{
		Class:     message.ClassFloatPoint,
		Size:      8,
		ByteOrder: message.OrderLE,
	}

	// 2 float64 values: 1.5, 2.5
	data := []byte{
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x3f, // 1.5
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x40, // 2.5
	}

	var result []float64
	err := Convert(dt, data, 2, &result)
	if err != nil {
		t.Fatalf("Convert failed: %v", err)
	}

	if len(result) != 2 {
		t.Fatalf("expected 2 elements, got %d", len(result))
	}

	if result[0] != 1.5 {
		t.Errorf("result[0] = %f, want 1.5", result[0])
	}
	if result[1] != 2.5 {
		t.Errorf("result[1] = %f, want 2.5", result[1])
	}
}

func TestConvertString(t *testing.T) {
	dt := &message.Datatype{
		Class:         message.ClassString,
		Size:          10,
		StringPadding: message.PadNullTerm,
	}

	// 2 null-terminated strings, 10 bytes each
	data := []byte{
		'h', 'e', 'l', 'l', 'o', 0, 0, 0, 0, 0,
		'w', 'o', 'r', 'l', 'd', 0, 0, 0, 0, 0,
	}

	var result []string
	err := Convert(dt, data, 2, &result)
	if err != nil {
		t.Fatalf("Convert failed: %v", err)
	}

	if len(result) != 2 {
		t.Fatalf("expected 2 elements, got %d", len(result))
	}

	if result[0] != "hello" {
		t.Errorf("result[0] = %q, want \"hello\"", result[0])
	}
	if result[1] != "world" {
		t.Errorf("result[1] = %q, want \"world\"", result[1])
	}
}

func TestExportName(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"name", "Name"},
		{"Name", "Name"},
		{"my_field", "My_field"},
		{"123abc", "123abc"},
		{"field-name", "Field_name"},
		{"", "Field"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got := exportName(tt.input)
			if got != tt.expected {
				t.Errorf("exportName(%q) = %q, want %q", tt.input, got, tt.expected)
			}
		})
	}
}

func TestByteOrder(t *testing.T) {
	dtLE := &message.Datatype{ByteOrder: message.OrderLE}
	dtBE := &message.Datatype{ByteOrder: message.OrderBE}

	if ByteOrder(dtLE).String() != "LittleEndian" {
		t.Error("expected LittleEndian")
	}
	if ByteOrder(dtBE).String() != "BigEndian" {
		t.Error("expected BigEndian")
	}
}

func TestIsNumeric(t *testing.T) {
	intDt := &message.Datatype{Class: message.ClassFixedPoint}
	floatDt := &message.Datatype{Class: message.ClassFloatPoint}
	stringDt := &message.Datatype{Class: message.ClassString}

	if !IsNumeric(intDt) {
		t.Error("int should be numeric")
	}
	if !IsNumeric(floatDt) {
		t.Error("float should be numeric")
	}
	if IsNumeric(stringDt) {
		t.Error("string should not be numeric")
	}
}
