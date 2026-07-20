package hdf5

import (
    "testing"
    "github.com/huangzhengshun/go-hdf5/internal/message"
)

func TestDebugDeepChain(t *testing.T) {
    path := skipIfNoTestdata(t, "deep_chain.h5")

    f, err := Open(path)
    if err != nil {
        t.Fatalf("Open failed: %v", err)
    }
    defer f.Close()

    root := f.Root()

    // Print all messages in root header
    allMsgs := root.header.Messages
    t.Logf("Total messages: %d", len(allMsgs))
    for i, msg := range allMsgs {
        t.Logf("  [%d] Type=%d (%T)", i, msg.Type(), msg)
    }

    // Print Link messages
    links := root.header.GetMessages(message.TypeLink)
    t.Logf("Found %d Link messages", len(links))

    // Check for symbol table
    symMsg := root.header.GetMessage(message.TypeSymbolTable)
    if symMsg != nil {
        t.Log("Symbol table present!")
        st := symMsg.(*message.SymbolTable)
        t.Logf("  BTree addr: %x, LocalHeap addr: %x", st.BTreeAddress, st.LocalHeapAddress)
    } else {
        t.Log("No symbol table")
    }

    // Check for link info message
    liMsg := root.header.GetMessage(message.TypeLinkInfo)
    if liMsg != nil {
        t.Logf("Link Info present: %T", liMsg)
    } else {
        t.Log("No link info")
    }

    members, err := root.Members()
    if err != nil {
        t.Fatalf("Members failed: %v", err)
    }
    t.Logf("Total members: %d", len(members))
}
