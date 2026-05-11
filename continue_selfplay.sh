#!/usr/bin/env bash
# Continue self-play training where previous cycles left off, visualizing
# each new checkpoint when it lands.
#
# Auto-detects the highest existing cycle under runs/sp-{evader,pursuer}-* and
# alternates roles from there: the just-trained role becomes the next cycle's
# opponent. Numbering is strictly increasing across both roles.
#
# Usage:
#   ./continue_selfplay.sh                          # default: 4 more cycles, 30s view each
#   ./continue_selfplay.sh -n 8                     # 8 more cycles
#   ./continue_selfplay.sh -n 4 --no-view           # skip the viewer between cycles
#   ./continue_selfplay.sh -n 4 --view-secs 90      # custom viewer duration
#   ./continue_selfplay.sh --updates 3000           # longer per-cycle training
set -euo pipefail

cd "$(dirname "$0")"

NCYCLES=4
VIEW=1
VIEW_SECS=30
UPDATES=1500
N_ENVS=1024
N_STEPS=128
MINIBATCH=8192
LR=3e-5
CLIP=0.1
TARGET_KL=0.02
ENT_COEF=0.001

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n)           NCYCLES="$2"; shift 2 ;;
    --no-view)    VIEW=0; shift ;;
    --view-secs)  VIEW_SECS="$2"; shift 2 ;;
    --updates)    UPDATES="$2"; shift 2 ;;
    --n-envs)     N_ENVS="$2"; shift 2 ;;
    --lr)         LR="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,18p' "$0"; exit 0 ;;
    *)
      echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

# Highest cycle number for a given role, or empty if no runs exist.
latest_cycle() {
  local role=$1
  ls -1d runs/sp-"${role}"-* 2>/dev/null \
    | sed -E "s|.*/sp-${role}-([0-9]+).*|\\1|" \
    | sort -n | tail -1
}

last_p=$(latest_cycle pursuer)
last_e=$(latest_cycle evader)
last_p=${last_p:-}
last_e=${last_e:-}

if [[ -z "$last_p" && -z "$last_e" ]]; then
  echo "ERR: no existing self-play runs under runs/sp-*. Train cycle 0 first:" >&2
  echo "     poetry run python train_selfplay.py --role pursuer \\" >&2
  echo "       --opp-model runs/pursuit-lidar/model.pkl --name sp-pursuer-0" >&2
  exit 1
fi

p_num=${last_p:-0}
e_num=${last_e:-0}

# The fresher of the two checkpoints is the strongest opponent — train the
# OTHER role against it. Next numbering = max(both) + 1.
if (( p_num >= e_num )); then
  next_role=evader
  opp_role=pursuer
  opp_num=$p_num
else
  next_role=pursuer
  opp_role=evader
  opp_num=$e_num
fi
next=$(( (p_num > e_num ? p_num : e_num) + 1 ))

echo "[continue_selfplay] starting from sp-pursuer-${p_num:-none} / sp-evader-${e_num:-none}"
echo "[continue_selfplay] running $NCYCLES cycles, ${UPDATES} updates each"

for ((i=0; i<NCYCLES; i++)); do
  opp_path="runs/sp-${opp_role}-${opp_num}/model.pkl"
  out_name="sp-${next_role}-${next}"
  if [[ ! -f "$opp_path" ]]; then
    echo "ERR: opponent checkpoint missing: $opp_path" >&2
    exit 1
  fi

  echo
  echo "==== CYCLE ${next}: train ${next_role} vs sp-${opp_role}-${opp_num} ===="
  poetry run python train_selfplay.py \
    --role "$next_role" \
    --opp-model "$opp_path" \
    --updates "$UPDATES" \
    --n-envs "$N_ENVS" \
    --n-steps "$N_STEPS" \
    --minibatch "$MINIBATCH" \
    --lr "$LR" \
    --clip "$CLIP" \
    --target-kl "$TARGET_KL" \
    --ent-coef "$ENT_COEF" \
    --hidden 256 256 \
    --name "$out_name"

  if (( VIEW )); then
    echo "[continue_selfplay] viewer for ${out_name} (${VIEW_SECS}s, Esc closes early)"
    # timeout sends SIGTERM at the deadline; pygame cleans up. Don't propagate
    # a non-zero exit from timeout — that's expected when the watchdog fires.
    timeout --foreground "${VIEW_SECS}s" \
      poetry run python viewer.py --run "runs/${out_name}" || true
  fi

  # Roll roles: the just-trained model becomes next cycle's opponent.
  opp_role=$next_role
  opp_num=$next
  next_role=$( [[ $next_role == evader ]] && echo pursuer || echo evader )
  next=$((next + 1))
done

echo
echo "[continue_selfplay] done. Latest checkpoints:"
ls -1dt runs/sp-*-* 2>/dev/null | head -4
