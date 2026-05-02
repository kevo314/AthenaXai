#!/bin/bash
###############################################################################
# Athena Xai v1 — memory_diag.sh
# Shows detailed memory state: physical RAM, swap, fragmentation, CUDA,
# and contiguous block availability. Run between service startups (and
# during runtime turns) to diagnose cudaMalloc failures on Jetson unified
# memory.
#
# Captures both host-side and (via the orchestrator) in-container snapshots:
#   - Warm-idle snapshot when Athena finishes booting (after greeting plays)
#   - Per-turn snapshots throttled to every 15s during a session
#   - tegrastats lfb (largest contiguous GPU-allocatable block) on every snap
#   - /proc/meminfo key fields (MemAvailable, Cached, AnonPages, Slab, …)
#   - Container memory annotated, not double-counted (Jetson unified memory)
#   - debugfs auto-mount attempt so nvmap allocations become visible
#   - Headroom summary line: MemAvailable + GPU lfb (the honest numbers)
#
# Usage:
#   ~/athena/memory_diag.sh              # full report
#   ~/athena/memory_diag.sh --brief      # one-line summary
###############################################################################

BRIEF=false
[ "$1" = "--brief" ] && BRIEF=true

# ── 0. Try to mount debugfs so we can read nvmap allocations. Idempotent;
#       harmless if already mounted or if we don't have privileges. ──
if [ ! -d /sys/kernel/debug/nvmap ] && [ ! -f /sys/kernel/debug/extfrag/extfrag_index ]; then
    sudo mount -t debugfs debugfs /sys/kernel/debug 2>/dev/null || true
fi

echo "============================================"
echo "  Athena Xai v1 — Memory Diagnostic"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
echo ""

# ── 1. Basic memory ──
echo "── RAM ──"
free -m | awk '
/Mem:/   { printf "  Total: %s MB | Used: %s MB | Free: %s MB | Available: %s MB\n", $2, $3, $4, $7 }
/Swap:/  { printf "  Swap:  %s MB | Used: %s MB | Free: %s MB\n", $2, $3, $4 }
'
echo "  (HEADROOM = 'Available', not 'Free' — Available counts reclaimable cache)"
echo ""

# ── 1b. /proc/meminfo key fields ──
echo "── /proc/meminfo (key fields) ──"
if [ -f /proc/meminfo ]; then
    awk '/^(MemAvailable|Cached|Buffers|Mapped|Slab|SReclaimable|AnonPages|Shmem|Dirty|Writeback|KernelStack|PageTables|VmallocUsed|HugePages_Total|HardwareCorrupted):/ {
        printf "  %-20s %s %s\n", $1, $2, $3
    }' /proc/meminfo
else
    echo "  /proc/meminfo not available"
fi
echo ""

# ── 2. Swap devices ──
echo "── Swap Devices ──"
if swapon --show 2>/dev/null | grep -q .; then
    swapon --show 2>/dev/null | while read line; do echo "  $line"; done
else
    echo "  (no swap active)"
fi
echo ""

# Check for zram (uses RAM, doesn't help CUDA)
ZRAM_COUNT=$(cat /proc/swaps 2>/dev/null | grep -c zram)
if [ "$ZRAM_COUNT" -gt 0 ]; then
    echo "  !! WARNING: zram swap detected ($ZRAM_COUNT devices)"
    echo "     zram uses compressed RAM — does NOT free physical memory for CUDA"
    cat /proc/swaps 2>/dev/null | grep zram | while read line; do echo "     $line"; done
    echo ""
fi

# ── 3. VM tuning parameters ──
echo "── VM Settings ──"
echo "  swappiness:        $(cat /proc/sys/vm/swappiness 2>/dev/null || echo '?')"
echo "  vfs_cache_pressure: $(cat /proc/sys/vm/vfs_cache_pressure 2>/dev/null || echo '?')"
echo "  dirty_ratio:        $(cat /proc/sys/vm/dirty_ratio 2>/dev/null || echo '?')"
echo "  dirty_bg_ratio:     $(cat /proc/sys/vm/dirty_background_ratio 2>/dev/null || echo '?')"
echo "  min_free_kbytes:    $(cat /proc/sys/vm/min_free_kbytes 2>/dev/null || echo '?')"
echo ""

# ── 4. Buddy allocator — contiguous block availability ──
echo "── Contiguous Memory Blocks (buddyinfo) ──"
echo "  Each column is a power-of-2 block size (4KB, 8KB, ... 4MB)"
echo "  Higher-order blocks = larger contiguous allocations possible"
echo ""
if [ -f /proc/buddyinfo ]; then
    cat /proc/buddyinfo | while read line; do
        echo "  $line"
    done
    echo ""

    # Calculate largest contiguous block available, plus order-10 (4MB) sum
    echo "  Estimated largest contiguous blocks:"
    cat /proc/buddyinfo | while read node zone name o0 o1 o2 o3 o4 o5 o6 o7 o8 o9 o10 rest; do
        for order in 10 9 8 7 6; do
            eval count=\$o${order}
            if [ -n "$count" ] && [ "$count" -gt 0 ] 2>/dev/null; then
                size_kb=$((4 * (1 << order)))
                size_mb=$((size_kb / 1024))
                total_mb=$((size_mb * count))
                echo "    $name: ${count}x ${size_mb}MB blocks = ${total_mb}MB contiguous available (order $order)"
                break
            fi
        done
    done
    echo ""
    # 4MB block totals (order 10) per zone — what CUDA cares about most
    echo "  4MB-block (order-10) totals per zone:"
    cat /proc/buddyinfo | while read node zone name o0 o1 o2 o3 o4 o5 o6 o7 o8 o9 o10 rest; do
        c10=${o10:-0}
        mb=$((4 * c10))
        echo "    $name: ${c10}x 4MB = ${mb}MB"
    done
else
    echo "  /proc/buddyinfo not available"
fi
echo ""

# ── 5. Memory fragmentation index ──
echo "── Fragmentation ──"
if [ -f /proc/sys/vm/compact_memory ]; then
    if [ -f /sys/kernel/debug/extfrag/extfrag_index ]; then
        echo "  Extfrag index (0=no frag, 1=total frag):"
        cat /sys/kernel/debug/extfrag/extfrag_index 2>/dev/null | head -5 | while read line; do
            echo "    $line"
        done
    else
        echo "  (extfrag_index not available — debugfs not mounted or no privileges)"
    fi
    echo "  Compact memory: available (can trigger with 'echo 1 > /proc/sys/vm/compact_memory')"
else
    echo "  Memory compaction: not available"
fi
echo ""

# ── 6. CUDA / GPU memory ──
echo "── GPU Memory (CUDA / Tegra) ──"
# tegrastats single snapshot — pull lfb out into its own line for clarity.
if command -v tegrastats &>/dev/null; then
    echo "  tegrastats snapshot:"
    TS_LINE=$(timeout 2 tegrastats --interval 1000 2>/dev/null | head -1)
    if [ -n "$TS_LINE" ]; then
        echo "    $TS_LINE"
        # Parse lfb (largest free block) — Jetson reports it inside RAM (...)
        # Format example: "RAM 3175/7620MB (lfb 137x4MB)"
        LFB=$(echo "$TS_LINE" | grep -oE 'lfb [0-9]+x[0-9]+MB' | head -1)
        if [ -n "$LFB" ]; then
            # Extract count and block size
            LFB_COUNT=$(echo "$LFB" | grep -oE '[0-9]+x' | tr -d 'x')
            LFB_SIZE=$(echo "$LFB"  | grep -oE 'x[0-9]+MB' | tr -dc '0-9')
            LFB_TOTAL=$((LFB_COUNT * LFB_SIZE))
            echo "    GPU lfb: ${LFB_COUNT} x ${LFB_SIZE}MB = ${LFB_TOTAL}MB largest contiguous (CUDA-allocatable)"
        fi
    else
        echo "    (tegrastats produced no output)"
    fi
    echo ""
fi

# NVML / nvidia-smi (if available on Jetson)
if command -v nvidia-smi &>/dev/null; then
    echo "  nvidia-smi:"
    nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits 2>/dev/null | while read line; do
        echo "    $line"
    done
    echo ""
fi

# NvMap memory usage (Jetson-specific)
if [ -f /sys/kernel/debug/nvmap/iovmm/allocations ]; then
    NVMAP_TOTAL=$(awk '{sum+=$2} END {printf "%.0f", sum/1048576}' /sys/kernel/debug/nvmap/iovmm/allocations 2>/dev/null)
    echo "  NvMap iovmm allocations: ~${NVMAP_TOTAL}MB"
elif [ -f /sys/kernel/debug/nvmap/iovmm/clients ]; then
    echo "  NvMap iovmm clients:"
    head -20 /sys/kernel/debug/nvmap/iovmm/clients 2>/dev/null | sed 's/^/    /'
elif [ -d /sys/kernel/debug/nvmap ]; then
    echo "  NvMap debugfs available but allocation details not readable:"
    ls /sys/kernel/debug/nvmap 2>/dev/null | head -5 | sed 's/^/    /'
else
    echo "  NvMap debugfs: not available (mount attempt at top of script failed —"
    echo "                  may need root, or kernel built without nvmap debugfs)"
fi
echo ""

# ── 7. Docker container memory usage ──
echo "── Docker Container Memory ──"
echo "  NOTE: On Jetson unified memory, container MemUsage includes mmap'd"
echo "        model file pages. Those pages ALSO count toward system Cached"
echo "        and toward MemAvailable — do NOT add container totals to"
echo "        process RSS to estimate pressure. Trust 'Available' above."
sudo docker stats --no-stream --format "  {{.Name}}: {{.MemUsage}} ({{.MemPerc}})" 2>/dev/null | grep athena || echo "  (no athena containers running)"
echo ""

# ── 8. Top memory consumers ──
echo "── Top 10 Memory Consumers ──"
ps aux --sort=-%mem 2>/dev/null | head -11 | awk 'NR==1 {printf "  %-10s %-8s %-6s %s\n", "USER", "PID", "%MEM", "COMMAND"} NR>1 {printf "  %-10s %-8s %-6s %s\n", $1, $2, $4, $11}'
echo ""

# ── 9. Headroom summary (one-line, honest) ──
echo "── Headroom summary ──"
AVAIL=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null)
if [ -n "$AVAIL" ]; then
    AVAIL_MB=$((AVAIL / 1024))
    echo "  System MemAvailable: ${AVAIL_MB} MB  (this is what you can grow into)"
fi
if [ -n "$LFB_TOTAL" ]; then
    echo "  GPU lfb total:       ${LFB_TOTAL} MB  (largest CUDA-allocatable contiguous)"
fi
echo ""

echo "============================================"
echo "  Diagnostic complete"
echo "============================================"
