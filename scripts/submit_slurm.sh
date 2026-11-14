#!/usr/bin/env bash
# Submit one exclusive task per QE job, then collect the finished case.
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 /absolute/path/to/planned-case" >&2
    exit 2
fi
case_dir="$(realpath "$1")"
mapfile -t job_ids < <(python -m ag_atlas.batch "$case_dir" --list)
if (( ${#job_ids[@]} == 0 )); then
    echo "No jobs listed for $case_dir" >&2
    exit 2
fi
command -v pw.x >/dev/null
mkdir -p "$case_dir/slurm-logs"
printf -v work_command '%q ' python -m ag_atlas.batch "$case_dir" --index
array_id="$(sbatch --parsable --array="0-$(("${#job_ids[@]}" - 1))%${ATLAS_MAX_ACTIVE:-8}" \
    --ntasks="${ATLAS_MPI_TASKS:-8}" --cpus-per-task="${ATLAS_THREADS:-1}" \
    --mem="${ATLAS_MEMORY:-24G}" --time="${ATLAS_WALLTIME:-12:00:00}" \
    --job-name=ag-qe --output="$case_dir/slurm-logs/%A-%a.log" \
    --wrap="export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK; $work_command \"\$SLURM_ARRAY_TASK_ID\" --mpi srun --threads \"\$SLURM_CPUS_PER_TASK\"")"
printf -v collect_command '%q ' ag-atlas collect "$case_dir"
sbatch --dependency="afterany:${array_id%%;*}" --ntasks=1 --mem=2G \
    --time=00:30:00 --job-name=ag-collect --output="$case_dir/slurm-logs/%j-collect.log" \
    --wrap="$collect_command"
printf 'QE array %s\n' "$array_id"
