#!/usr/bin/env python3
"""
patch_hunav_agent_manager.py -- lets the bridge change a HuNav agent's goals
and walking speed WHILE the simulation runs (local patch, 2026-10-05).

WHY
hunav_agent_manager reads goals and speed only once, from the yaml, when the
agents are initialised. In AgentManager::updateAgents() (agent_manager.cpp)
the block that would take new goals from each /compute_agents request is
commented out upstream, so whatever the bridge sends is ignored.

WHAT THE PATCH DOES (one block inserted in updateAgents, nothing removed)
  * goals: the bridge normally sends back the goal list of the previous
    response, i.e. the manager's own list -> no change, HuNav keeps cycling
    through its goals exactly as before. Only when the bridge sends a
    DIFFERENT list does it replace the agent's goals (goal radius kept).
  * speed: a desired_velocity > 0 that differs from the current configured
    value replaces it (the responses carry 0, which means "leave it").
So with the unchanged hunav_model_bridge.py nothing behaves differently.

USAGE
    python3 patch_hunav_agent_manager.py <path to hunav_agent_manager/src/agent_manager.cpp>
then rebuild:  cd ~/ros2_ws && colcon build --packages-select hunav_agent_manager
A backup agent_manager.cpp.orig is written next to the file first. Running it
twice does nothing the second time. To undo: copy the .orig back and rebuild.
"""
import shutil
import sys
from pathlib import Path

MARK = 'LOCAL PATCH (thesis, 2026-10-05)'
ANCHOR = '    // update closest obstacles\n'
FUNC = 'bool AgentManager::updateAgents('

BLOCK = '''    // ---- LOCAL PATCH (thesis, 2026-10-05): runtime goal / speed control ----
    // The caller normally echoes the goals of the previous response (this
    // manager's own list), so nothing changes. A DIFFERENT, non-empty list
    // replaces the agent's goals (goal radius kept). A desired_velocity > 0
    // that differs from the configured one replaces it (0 = keep).
    {
      auto & sa = agents_[a.id].sfmAgent;
      bool differ = (a.goals.size() != sa.goals.size());
      if (!differ)
      {
        auto it = sa.goals.begin();
        for (const auto & g : a.goals)
        {
          if (std::fabs(g.position.x - it->center.getX()) > 1e-6 ||
              std::fabs(g.position.y - it->center.getY()) > 1e-6)
          {
            differ = true;
            break;
          }
          ++it;
        }
      }
      if (!a.goals.empty() && differ)
      {
        const double r = !sa.goals.empty() ? sa.goals.front().radius :
                         (a.goal_radius > 0.0 ? a.goal_radius : 0.3);
        sa.goals.clear();
        for (const auto & g : a.goals)
        {
          sfm::Goal sfmg;
          sfmg.center.setX(g.position.x);
          sfmg.center.setY(g.position.y);
          sfmg.radius = r;
          sa.goals.push_back(sfmg);
        }
        printf("[AgentManager] agent %i: %zu new goal(s) from the caller\\n", a.id, a.goals.size());
      }
      if (a.desired_velocity > 0.0 && std::fabs(a.desired_velocity - orig_desired_vels_[a.id]) > 1e-6)
      {
        orig_desired_vels_[a.id] = a.desired_velocity;
        sa.desiredVelocity = a.desired_velocity;
        printf("[AgentManager] agent %i: desired velocity set to %.2f m/s\\n", a.id, a.desired_velocity);
      }
    }
    // ---- end of LOCAL PATCH ----

'''


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    path = Path(sys.argv[1]).expanduser().resolve()
    src = path.read_text()
    if MARK in src:
        print(f'{path} is already patched. Nothing to do.')
        return
    start = src.find(FUNC)
    if start < 0:
        sys.exit(f'Could not find "{FUNC}" in {path}. Is this hunav_agent_manager/src/agent_manager.cpp?')
    end = src.find('\n}\n', start)          # end of updateAgents()
    pos = src.find(ANCHOR, start)
    if pos < 0 or (end > 0 and pos > end):
        sys.exit(f'Could not find the line "{ANCHOR.strip()}" inside updateAgents(). Your HuNav version differs '
                 f'from the one this patch was written for: send me updateAgents() from {path} and I will adapt it.')
    new = src[:pos] + BLOCK + src[pos:]
    if '#include <cmath>' not in new:
        new = '#include <cmath>\n' + new
    backup = path.with_name(path.name + '.orig')
    shutil.copy2(path, backup)
    path.write_text(new)
    print(f'Patched {path}\n  backup: {backup}\n'
          f'Now rebuild:  cd ~/ros2_ws && colcon build --packages-select hunav_agent_manager && source install/setup.bash')


if __name__ == '__main__':
    main()