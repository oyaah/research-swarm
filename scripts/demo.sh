#!/usr/bin/env bash
# Demo queries for Research Swarm — portfolio showcase
# These queries are optimized for impressive streaming TUI output + portfolio mode
#
# Usage:
#   ./scripts/demo.sh                    # interactive picker
#   ./scripts/demo.sh 1                  # run demo 1 directly
#   ./scripts/demo.sh portfolio          # run portfolio mode demo

set -euo pipefail
API="http://localhost:8000"

# Verify API is up
if ! curl -sf "${API}/healthz" >/dev/null 2>&1; then
  echo "Error: API not running. Start with: ./bootstrap.sh"
  exit 1
fi

DEMOS=(
  "What are the most significant AI coding assistants in 2025 and how do they compare for professional developers?"
  "What caused the collapse of Silicon Valley Bank and what regulatory changes followed?"
  "How does Retrieval Augmented Generation work, and what are the current best practices for production RAG systems?"
)

MODES=("single" "single" "portfolio")
DEPTHS=("standard" "deep" "deep")

show_menu() {
  echo ""
  echo "Research Swarm — Demo Queries"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  for i in "${!DEMOS[@]}"; do
    idx=$((i+1))
    echo "  $idx. [${MODES[$i]}/${DEPTHS[$i]}] ${DEMOS[$i]}"
  done
  echo ""
  echo "  p. Portfolio mode — parallel lanes + Pareto selection (demo 1)"
  echo "  q. Quit"
  echo ""
}

run_demo() {
  local query="$1"
  local mode="${2:-single}"
  local depth="${3:-standard}"

  echo ""
  echo "Starting: $query"
  echo "Mode: $mode | Depth: $depth"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  RESPONSE=$(curl -sf -X POST "${API}/v1/research" \
    -H "Content-Type: application/json" \
    -d "{\"query\": $(echo "$query" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))'), \"mode\": \"$mode\", \"depth\": \"$depth\", \"budget_mode\": \"balanced\"}")

  SESSION_ID=$(echo "$RESPONSE" | python3 -c "import json,sys; print(json.load(sys.stdin)['session_id'])")
  echo "Session: $SESSION_ID"
  echo ""

  # Stream events
  curl -sN "${API}/v1/stream/${SESSION_ID}" | while IFS= read -r line; do
    if [[ "$line" == data:* ]]; then
      payload="${line#data: }"
      # Pretty-print key events
      event_type=$(echo "$payload" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('event_type',''))" 2>/dev/null || echo "")
      case "$event_type" in
        plan_update)
          echo "[Planner] Research plan ready"
          ;;
        evidence_item)
          title=$(echo "$payload" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('payload',{}).get('title','')[:60])" 2>/dev/null || echo "")
          echo "[Researcher] Evidence: $title"
          ;;
        reasoning_summary)
          agent=$(echo "$payload" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('agent',''))" 2>/dev/null || echo "")
          summary=$(echo "$payload" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('payload',{}).get('summary','')[:80])" 2>/dev/null || echo "")
          [[ -n "$summary" ]] && echo "[$agent] $summary"
          ;;
        final_markdown)
          echo ""
          echo "[Writer] Report complete!"
          ;;
        done)
          echo ""
          echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
          echo "Report saved to artifacts/reports/$SESSION_ID/"
          break
          ;;
      esac
    fi
  done
}

# Handle direct argument
if [[ $# -gt 0 ]]; then
  case "$1" in
    1|2|3)
      idx=$((${1}-1))
      run_demo "${DEMOS[$idx]}" "${MODES[$idx]}" "${DEPTHS[$idx]}"
      ;;
    portfolio|p)
      run_demo "${DEMOS[0]}" "portfolio" "deep"
      ;;
    *)
      echo "Usage: $0 [1|2|3|portfolio]"
      exit 1
      ;;
  esac
  exit 0
fi

# Interactive menu
while true; do
  show_menu
  read -rp "Pick a demo [1-3/p/q]: " choice
  case "$choice" in
    1|2|3)
      idx=$((choice-1))
      run_demo "${DEMOS[$idx]}" "${MODES[$idx]}" "${DEPTHS[$idx]}"
      ;;
    p|portfolio)
      run_demo "${DEMOS[0]}" "portfolio" "deep"
      ;;
    q|quit)
      exit 0
      ;;
    *)
      echo "Invalid choice."
      ;;
  esac
done
