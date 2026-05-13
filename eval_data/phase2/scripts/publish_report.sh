#!/bin/bash
# Upload the report HTML to GCS for static hosting.
#
# Usage:
#   bash publish_report.sh                      # publish report/index.html
#   bash publish_report.sh path/to/file.html    # publish a different file
#
# Bucket: autoresearch-demo-report (set via BUCKET env var to override)
# Public URL: https://storage.googleapis.com/<BUCKET>/index.html
set -e

BUCKET="${BUCKET:-autoresearch-demo-report}"
SRC="${1:-$(dirname "$0")/../report/index.html}"

if [ ! -f "$SRC" ]; then
    echo "Error: $SRC not found. Run build_report.py first."
    exit 1
fi

DEST="gs://${BUCKET}/index.html"

echo "Uploading $SRC -> $DEST"
gcloud storage cp "$SRC" "$DEST" \
    --cache-control="public, max-age=300" \
    --content-type="text/html; charset=utf-8"

# Public read is granted at the BUCKET level (the bucket uses Uniform
# bucket-level access, which disables per-object ACLs). This is idempotent —
# re-running just confirms the existing grant.
echo "Ensuring public read on bucket $BUCKET"
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
    --member=allUsers --role=roles/storage.objectViewer >/dev/null 2>&1 || true

echo ""
echo "========================================"
echo "  Published: https://storage.googleapis.com/${BUCKET}/index.html"
echo "========================================"
