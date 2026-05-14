"""Diagnostic script: prints ALL modifier key events to help debug Fn detection."""
import time
import sys

sys.stdout.reconfigure(line_buffering=True)

print("=== Hotkey Diagnostic ===")
print("This will print ALL modifier key events (Fn, Shift, Cmd, etc.)")
print("Try pressing Fn, Shift, Cmd, Option, etc. and see what shows up.")
print("Press Ctrl+C to quit.\n")

try:
    from AppKit import (
        NSEvent,
        NSEventMaskFlagsChanged,
        NSEventModifierFlagFunction,
        NSRunLoop,
        NSDate,
    )
except ImportError:
    print("ERROR: AppKit not available")
    sys.exit(1)

event_count = 0

def handler(event):
    global event_count
    event_count += 1
    flags = event.modifierFlags()
    parts = []
    if flags & NSEventModifierFlagFunction:
        parts.append("Fn")
    if flags & 0x20000:
        parts.append("Shift")
    if flags & 0x40000:
        parts.append("Control")
    if flags & 0x80000:
        parts.append("Option")
    if flags & 0x100000:
        parts.append("Command")
    if flags & 0x10000:
        parts.append("CapsLock")

    active = ", ".join(parts) if parts else "(none)"
    print(f"  Event #{event_count}: flags=0x{flags:08x} active=[{active}] keyCode={event.keyCode()}")

monitor = NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
    NSEventMaskFlagsChanged, handler
)

if monitor is None:
    print("ERROR: addGlobalMonitor returned None.")
    print("You need to grant Accessibility permission to your terminal app.")
    print("  System Settings > Privacy & Security > Accessibility")
    sys.exit(1)

local = NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
    NSEventMaskFlagsChanged,
    lambda e: (handler(e), e)[1],
)

print(f"Monitor active. Waiting for key events...")
print(f"(If you see nothing when pressing keys, your terminal lacks Accessibility permission)\n")

loop = NSRunLoop.currentRunLoop()
try:
    while True:
        loop.runMode_beforeDate_("kCFRunLoopDefaultMode", NSDate.dateWithTimeIntervalSinceNow_(0.1))
except KeyboardInterrupt:
    print(f"\n\nTotal events captured: {event_count}")
    NSEvent.removeMonitor_(monitor)
    if local:
        NSEvent.removeMonitor_(local)
