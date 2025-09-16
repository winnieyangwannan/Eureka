⏺ For a humanoid "run as fast as possible" task, success would be measured by
  forward progress speed, not traditional binary success conditions.

  Looking at the humanoid task code, success is defined in compute_success() at line
   368:

  consecutive_successes = progress_reward.mean()

  Where progress_reward (line 361) is:
  progress_reward = potentials - prev_potentials

  And potentials represents negative distance to target (from observation code line
  302):
  potentials = -torch.norm(to_target, p=2, dim=-1) / dt

  Success criteria for humanoid running:

  1. Speed-based success: consecutive_successes measures the average forward 
  velocity toward the target (1000m away on x-axis)
  2. Higher values = faster running: More positive progress_reward means faster
  approach to target
  3. Continuous metric: Unlike discrete tasks, this is a continuous speed
  measurement rather than binary success/failure
  4. Termination conditions: Task fails if torso height drops below
  termination_height (falling down)

  So for "run as fast as possible," success = sustained high forward velocity
  without falling, measured as average progress toward the distant target over
  consecutive timesteps.

---
In the humanoid task, potentials is a distance-based reward shaping mechanism that
   measures progress toward the goal.

  From the code:

  potentials = -torch.norm(to_target, p=2, dim=-1) / dt

  Key properties:

  1. Distance to target: torch.norm(to_target, p=2, dim=-1) calculates Euclidean
  distance to target
  2. Negative distance: The negative sign means closer to target = higher potential
  3. Time-normalized: Divided by dt (timestep) to make it velocity-like
  4. Target location: Target is at [1000, 0, 0] - 1000 meters ahead in x-direction

  How it works:
  - Initial potential: -1000/dt (far from target)
  - As robot moves forward: Distance decreases, potential increases (becomes less
  negative)
  - Progress reward: potentials - prev_potentials gives the change in potential =
  forward velocity

  Example:
  - Robot at x=0: potential = -1000/dt
  - Robot moves to x=10: potential = -990/dt
  - Progress reward = -990/dt - (-1000/dt) = 10/dt = forward velocity

  Purpose: Provides dense reward signal proportional to forward movement speed,
  encouraging the humanoid to run toward the distant target rather than just staying
   alive.