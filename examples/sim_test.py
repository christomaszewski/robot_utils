import robot_primitives as rp

from context import robot_utils as rut

domain_width = 10.
max_flow_speed = -1.
boat_speed = 2.

domain = rp.areas.Domain.from_box_corners((0,0), (domain_width, domain_width))
flow_field = rp.fields.VectorField.from_uniform_vector((0.,max_flow_speed))

sim = rut.sim.SimpleSimulator(flow_field)

test_path = rp.paths.ConstrainedPath([(5.,2.), (5.,8.)])
traj, wp_times = sim.simulate_path(test_path, boat_speed)
print('Desired Output: (6, [5,8]), [0, 6]', traj[-1], wp_times)

test_path = rp.paths.ConstrainedPath([(5.,8.), (5.,2.)])
traj, wp_times = sim.simulate_path(test_path, boat_speed)
print('Desired Output: (2, [5,2]), [0, 2]', traj[-1], wp_times)

test_path = rp.paths.ConstrainedPath([(8.,5.), (2.,2.)])
traj, wp_times = sim.simulate_path(test_path, boat_speed)
print('Desired Output: (3, [2,2]), [0, 3]', traj[-1], wp_times)

test_path = rp.paths.ConstrainedPath([(2.,2.), (8.,5)])
traj, wp_times = sim.simulate_path(test_path, boat_speed)
print('Desired Output: (5, [8,5]), [0, 5]', traj[-1], wp_times)