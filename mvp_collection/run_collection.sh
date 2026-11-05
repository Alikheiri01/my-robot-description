#!/usr/bin/env bash
# run_collection.sh -- the unattended collection: runs the episode runner in BATCHES and restarts
# Gazebo + the whole launch between batches, so a long run never works on a simulation that has
# been up for hours (the rehearsal got slower with time: spawn timeouts, leftover models).
#
# USAGE (from mvp_collection/, in a terminal where ROS is sourced, nothing else running; call it as
#     'bash run_collection.sh', NOT with a full path: cleanup_hunav.sh kills every process whose command
#     line contains my_robot_description/mvp_collection/, this script included):
#     bash run_collection.sh datasets/mvp_v1/A_detour datasets/mvp_v1/B_corner ...
#     BATCH=3 bash run_collection.sh datasets/rehearsal/*        # environments per Gazebo session (default 4)
# Already recorded environments are skipped, so you can stop it (Ctrl+C) and start it again.
# Failed environments are tried again in the next batch (fresh Gazebo). It stops when everything is
# recorded, or after 2 batches in a row in which no environment worked.
# Logs: logs_collection/<time>_round<N>_launch.log (launch) and ..._runner.log (runner output).

BATCH="${BATCH:-4}"
LAUNCH_ARGS="${LAUNCH_ARGS:-gui:=false rviz:=false crop_view:=false}"
READY_WAIT_SEC=180
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE" || exit 1
LOGDIR="$HERE/logs_collection"
mkdir -p "$LOGDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

if [ "$#" -eq 0 ]; then
    echo "Give the scenario or environment folders, e.g.  bash run_collection.sh datasets/mvp_v1/A_detour datasets/mvp_v1/B_corner"
    exit 1
fi

# Where the cleanup script is: CLEANUP=/path bash run_collection.sh ..., else ~/cleanup_hunav.sh, else next to my_robot_description/
CLEANUP="${CLEANUP:-}"
if [ -z "$CLEANUP" ]; then
    for c in "$HOME/cleanup_hunav.sh" "$HERE/../cleanup_hunav.sh"; do
        if [ -f "$c" ]; then CLEANUP="$c"; break; fi
    done
fi
if [ -z "$CLEANUP" ] || [ ! -f "$CLEANUP" ]; then
    echo "ERROR: cleanup_hunav.sh not found (looked in $HOME and $HERE/..). Give it with CLEANUP=/path/cleanup_hunav.sh"
    exit 1
fi
echo "cleanup script: $CLEANUP"

LEFTOVER_PATTERN='ign gazebo|gzserver|gzclient|parameter_bridge|rviz2|hunav_|--ros-args|robot_driver|record_session|episode_runner'

# Stops the simulation AND checks that nothing is left.
# Round 2 of the first try started on top of leftovers from round 1 (two simulations at once:
# robot and pedestrian poses disagreed, every teleport failed), so this repeats until nothing is left
# and refuses to go on otherwise.
stop_all() {
    local k
    for k in 1 2 3 4 5; do
        bash "$CLEANUP" > /dev/null 2>&1
        ros2 daemon stop > /dev/null 2>&1
        sleep 3
        if ! pgrep -f -- "$LEFTOVER_PATTERN" > /dev/null; then return 0; fi
    done
    echo "ERROR: simulation processes are still running after 5 cleanups:"
    pgrep -af -- "$LEFTOVER_PATTERN" | head -20
    return 1
}
trap 'echo; echo "Ctrl+C: stopping everything..."; stop_all; exit 130' INT TERM
# Stop with Ctrl+C (Ctrl+Shift+C in your terminal setup), never Ctrl+Z: a suspended script leaves the simulation running.

round=0
no_progress=0
while true; do
    round=$((round + 1))
    echo "=== round $round: starting a fresh simulation ($(date +%H:%M:%S)) ==="
    if ! stop_all; then echo "Stopping: could not clean up. Close the leftovers by hand (see above), then start again."; exit 2; fi
    # shellcheck disable=SC2086
    setsid ros2 launch my_robot_description my_robot_hunav_full.launch.py $LAUNCH_ARGS \
        > "$LOGDIR/${STAMP}_round${round}_launch.log" 2>&1 &
    # Wait until the simulation is up: the ROS topic /odom exists and Gazebo answers. Time based, with progress
    # lines. (The runner itself then waits up to 90 s for the pedestrian pose and /odom messages, which is the
    # real readiness test; a stricter check here, on names inside the scene, never came true and cost 3 minutes.)
    ready=0
    t0=$SECONDS
    last_msg=0
    while [ $((SECONDS - t0)) -lt "$READY_WAIT_SEC" ]; do
        if timeout 10 ros2 topic list 2>/dev/null | grep -qx "/odom" && \
           timeout 10 ign service -s /world/empty/scene/info --reqtype ignition.msgs.Empty \
               --reptype ignition.msgs.Scene --timeout 5000 --req '' > /dev/null 2>&1; then
            ready=1; break
        fi
        if [ $((SECONDS - t0 - last_msg)) -ge 15 ]; then
            last_msg=$((SECONDS - t0)); echo "  waiting for the simulation to come up... ${last_msg} s"
        fi
        sleep 2
    done
    if [ "$ready" -ne 1 ]; then
        echo "The simulation did not come up within ${READY_WAIT_SEC} s (see $LOGDIR/${STAMP}_round${round}_launch.log)."
        no_progress=$((no_progress + 1))
        stop_all
        if [ "$no_progress" -ge 2 ]; then echo "Stopping: the simulation failed to start twice."; exit 2; fi
        continue
    fi
    echo "  simulation is up after $((SECONDS - t0)) s; letting it settle for 45 s"
    sleep 45
    python3 episode_runner.py "$@" --skip-done --max-envs "$BATCH" 2>&1 | tee "$LOGDIR/${STAMP}_round${round}_runner.log"
    rc=${PIPESTATUS[0]}
    stop_all || { echo "Stopping: could not clean up after the batch."; exit 2; }
    echo "=== round $round finished, runner exit code $rc ==="
    case "$rc" in
        0)   echo "ALL DONE."; exit 0 ;;
        3)   no_progress=0 ;;
        2)   no_progress=$((no_progress + 1))
             if [ "$no_progress" -ge 2 ]; then echo "Stopping: 2 batches in a row without a working environment."; exit 2; fi ;;
        130) echo "Stopped by Ctrl+C."; exit 130 ;;
        *)   echo "Runner ended with code $rc (not a normal end). Stopping."; exit "$rc" ;;
    esac
done