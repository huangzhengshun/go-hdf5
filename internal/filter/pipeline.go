package filter

import (
	"fmt"

	"github.com/huangzhengshun/go-hdf5/internal/message"
)

// Pipeline represents a filter pipeline that can decode chunk data.
type Pipeline struct {
	filters []Filter
}

// NewPipeline creates a filter pipeline from a FilterPipeline message.
func NewPipeline(fp *message.FilterPipeline) (*Pipeline, error) {
	if fp == nil || len(fp.Filters) == 0 {
		return &Pipeline{}, nil
	}

	p := &Pipeline{
		filters: make([]Filter, 0, len(fp.Filters)),
	}

	for _, info := range fp.Filters {
		f, err := New(info)
		if err != nil {
			return nil, fmt.Errorf("creating filter %d: %w", info.ID, err)
		}
		if f != nil {
			p.filters = append(p.filters, f)
		}
	}

	return p, nil
}

// Decode applies the filter pipeline to encoded data.
// The filterMask specifies which filters to skip (bit i = skip filter i).
// Filters are applied in reverse order (last filter first).
func (p *Pipeline) Decode(input []byte, filterMask uint32) ([]byte, error) {
	if len(p.filters) == 0 {
		return input, nil
	}

	data := input

	// Apply filters in reverse order
	for i := len(p.filters) - 1; i >= 0; i-- {
		// Check if this filter should be skipped
		if filterMask&(1<<uint(i)) != 0 {
			continue
		}

		var err error
		data, err = p.filters[i].Decode(data)
		if err != nil {
			return nil, fmt.Errorf("filter %d decode: %w", p.filters[i].ID(), err)
		}
	}

	return data, nil
}

// Empty returns true if the pipeline has no filters.
func (p *Pipeline) Empty() bool {
	return len(p.filters) == 0
}

// Len returns the number of filters in the pipeline.
func (p *Pipeline) Len() int {
	return len(p.filters)
}
