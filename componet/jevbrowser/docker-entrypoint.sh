#!/bin/sh
set -eu

chrome_profile="${CHROME_USER_DATA_DIR:-/tmp/jev-chrome}"
mkdir -p "$chrome_profile"

google-chrome \
    --headless=new \
    --no-sandbox \
    --disable-dev-shm-usage \
    --disable-gpu \
    --remote-debugging-address=127.0.0.1 \
    --remote-debugging-port=9222 \
    --user-data-dir="$chrome_profile" \
    about:blank >/tmp/chrome.log 2>&1 &
chrome_pid=$!

cleanup() {
    kill "$chrome_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

attempt=0
until curl --fail --silent http://127.0.0.1:9222/json/version >/dev/null; do
    attempt=$((attempt + 1))
    if ! kill -0 "$chrome_pid" 2>/dev/null; then
        cat /tmp/chrome.log >&2
        echo "Chrome exited before its debugging endpoint became ready." >&2
        exit 1
    fi
    if [ "$attempt" -ge 100 ]; then
        cat /tmp/chrome.log >&2
        echo "Timed out waiting for Chrome's debugging endpoint." >&2
        exit 1
    fi
    sleep 0.1
done

"$@" &
app_pid=$!
wait "$app_pid"
status=$?
exit "$status"
