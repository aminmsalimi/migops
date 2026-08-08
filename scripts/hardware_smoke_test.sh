#!/usr/bin/env bash
set -u

GPU="${1:-0}"

echo "== MIGOps hardware smoke test (read-only) =="
echo "GPU selector: ${GPU}"
echo

migops --version
echo

migops doctor || true
migops status || true
migops profiles --gpu "${GPU}" || true
migops users --gpu "${GPU}" || true

echo
echo "Smart Split probes:"
migops split --gpu "${GPU}" --instances 2 || true
migops split --gpu "${GPU}" --instances 4 || true

echo
echo "No changes were requested by this script."
