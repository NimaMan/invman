#!/usr/bin/env bash
# Run ALL full-budget >=5-optimizer-seed audits sequentially under the ~4-core cap.
#
# OBJECTIVE
# ---------
# Produce the real (non-smoke) seed-robust artifacts for every seeds-mode problem
# that currently lacks one, per invman/optimizer_seed_robustness_policy.py:
#   perishable_inventory, general_backorder_fixed_cost, ameliorating_inventory,
#   production_assembly_distribution_network (serial_case3 + pure_assembly = the
#   paper's pending M2 audit, + mixed re-audit), dual_sourcing, multi_echelon_serial.
#
# ALGORITHM
# ---------
# 1. Steps run SEQUENTIALLY, shortest-first (early artifacts, early signal); each
#    step is capped at 2 Rayon/OMP threads + mp_num_processors 2 (so peak ~2-4
#    cores, never the ~27-core default).
# 2. Each step logs to outputs/seed_robust_queue_logs/<step>.log; a failure does
#    NOT abort the queue (set +e per step) -- exit codes are collected.
# 3. Final step re-runs paper/generate_results_tables.py, whose fail-loud gate
#    checks the dual-sourcing artifact; the queue's last lines summarize every
#    step's exit code. Overall exit = 0 only if all steps passed.
#
# USAGE:  nohup bash scripts/run_all_seed_robust_full_budget_audits.sh &
set -u
cd "$(dirname "$0")/.."
REPO="$PWD"
LOGDIR="$REPO/outputs/seed_robust_queue_logs"
mkdir -p "$LOGDIR"
export RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH="$REPO"
SEEDS="9001 9002 9003 9004 9005"
declare -A CODES

run_step () {
  local name="$1"; shift
  echo "[$(date +%H:%M:%S)] START $name"
  "$@" >"$LOGDIR/$name.log" 2>&1
  CODES[$name]=$?
  echo "[$(date +%H:%M:%S)] END   $name exit=${CODES[$name]}"
}

run_step perishable python3 scripts/perishable_inventory/seed_robust_perishable_inventory.py \
  --budget full --seeds $SEEDS --mp_num_processors 2

run_step general_backorder python3 scripts/general_backorder_fixed_cost/seed_robust_general_backorder.py \
  --budget full --seeds $SEEDS --mp_num_processors 2

run_step ameliorating python3 scripts/ameliorating_inventory/seed_robust_ameliorating_inventory.py \
  --instance spirits_0001 --budget full --seeds $SEEDS --mp_num_processors 2

run_step padn_serial_case3 python3 scripts/production_assembly_distribution_network/seed_robust_serial_and_pure_assembly_networks.py \
  --topology serial_case3 --budget full --seeds $SEEDS --mp_num_processors 2

run_step padn_pure_assembly python3 scripts/production_assembly_distribution_network/seed_robust_serial_and_pure_assembly_networks.py \
  --topology pure_assembly --budget full --seeds $SEEDS --mp_num_processors 2

run_step padn_mixed python3 scripts/production_assembly_distribution_network/seed_robust_mixed_distribution_assembly_network.py \
  --budget full --depth 2 --sigma_init 0.2 --seeds $SEEDS --mp_num_processors 2 \
  --description "srp-standardized 5-seed mixed audit"

run_step dual_sourcing python3 scripts/dual_sourcing/seed_robust_warmstart_soft_tree_vs_cdi_paired_crn.py \
  --budget full --seeds $SEEDS --mp_num_processors 2

run_step serial_clark_scarf python3 scripts/multi_echelon_serial/seed_robust_multi_echelon_serial.py \
  --instance snyder_shen_example_6_1 --budget full --seeds $SEEDS --mp_num_processors 2

run_step table_gate python3 paper/generate_results_tables.py

echo "================ QUEUE SUMMARY ================"
FAIL=0
for k in perishable general_backorder ameliorating padn_serial_case3 padn_pure_assembly padn_mixed dual_sourcing serial_clark_scarf table_gate; do
  echo "  $k: exit=${CODES[$k]:-SKIPPED}"
  [ "${CODES[$k]:-1}" -ne 0 ] && FAIL=1
done
exit $FAIL
