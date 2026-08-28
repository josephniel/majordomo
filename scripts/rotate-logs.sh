#!/usr/bin/env bash
#
# Rotate the LaunchAgent log files in logs/.
#
# COPY-TRUNCATE, deliberately. launchd opens StandardOutPath/StandardErrorPath
# itself and holds those descriptors for the life of the bot, in append mode.
# Renaming the file would not give the running process a new one — it would keep
# writing into the renamed inode until the agent is restarted, so the "rotated"
# archive keeps growing and the live log stays empty. Copying the contents out
# and truncating in place keeps the same inode, so the open descriptor survives
# and the next write lands at offset 0.
#
# The window between cp and truncate can drop a line or two. That is the
# standard copy-truncate trade and is the reason this runs monthly at 04:00
# rather than during the working day.
#
# Run by the logrotate LaunchAgent (see deploy/logrotate.plist.example), or by
# hand with ./manage logrotate-run.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_DIR/logs"
ARCHIVE_DIR="$LOG_DIR/archive"

# Months of archives to keep. Six months of gzipped scheduler chatter is a few
# MB; the point of the cap is that this never becomes the thing that fills the
# disk it was added to protect.
KEEP_MONTHS="${MAJORDOMO_LOG_KEEP_MONTHS:-6}"

# Files smaller than this are left alone — rotating a 122-byte out.log every
# month just litters the archive.
MIN_BYTES="${MAJORDOMO_LOG_MIN_BYTES:-4096}"

[[ -d "$LOG_DIR" ]] || { echo "[logrotate] no logs/ directory — nothing to do"; exit 0; }
mkdir -p "$ARCHIVE_DIR"

stamp="$(date +%Y-%m-%d)"
rotated=0

for log in "$LOG_DIR"/*.log; do
    [[ -f "$log" ]] || continue

    size="$(stat -f%z "$log")"
    if [[ "$size" -lt "$MIN_BYTES" ]]; then
        continue
    fi

    base="$(basename "$log" .log)"
    archive="$ARCHIVE_DIR/$base.$stamp.log.gz"
    # A second run on the same day must not clobber the first.
    n=2
    while [[ -e "$archive" ]]; do
        archive="$ARCHIVE_DIR/$base.$stamp-$n.log.gz"
        n=$((n + 1))
    done

    # gzip from a pipe so the plaintext copy never touches the disk.
    if ! gzip -c "$log" > "$archive"; then
        echo "[logrotate] FAILED to archive $log — leaving it untouched" >&2
        rm -f "$archive"
        continue
    fi
    # Only now is it safe to drop the contents. `:` writes nothing and the
    # redirect truncates; the inode, and every fd pointing at it, survives.
    : > "$log"

    echo "[logrotate] $base.log ($(du -h "$archive" | cut -f1) gzipped) -> $(basename "$archive")"
    rotated=$((rotated + 1))
done

# Prune archives older than the retention window. -mtime is in days; months are
# approximated at 31 so a 6-month window never expires something 5 months old.
pruned="$(find "$ARCHIVE_DIR" -name '*.log.gz' -type f -mtime "+$((KEEP_MONTHS * 31))" -print -delete | wc -l | tr -d ' ')"

echo "[logrotate] rotated $rotated file(s), pruned $pruned archive(s) older than ${KEEP_MONTHS} months"
