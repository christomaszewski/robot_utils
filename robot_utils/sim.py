import math
import numpy as np

import robot_primitives as rp

class SimpleSimulator(object):
	""" A simple simulator that can compute total travel time along a path through
		 a flow field assuming a constant nominal velocity and instantaneous changes
		 of direction"""

	def __init__(self, flow_field):

		self._flow_field = flow_field
		self._dt = 0.0001 #s
		self._eps = 0.001 #m

	def simulate_path(self, path, boat_speed):
		# Instantiate starting position and time
		curr_time = 0.
		curr_pos = np.asarray(path[0])

		# Instantiate arrays to record useful stuff during sim
		wp_arrival_times = np.zeros(path.size)
		output = [(curr_time, curr_pos)]

		# Set first target along path
		target_idx = 1
		curr_target = np.asarray(path[target_idx])

		# Run sim until last point in path has been reached
		while target_idx < path.size:
			flow_vec = np.asarray(self._flow_field[tuple(curr_pos)])
			flow_speed = np.linalg.norm(flow_vec)

			target_vec = curr_target - curr_pos

			# Check if we've reached the target waypoint
			if np.linalg.norm(target_vec) < self._eps:
				wp_arrival_times[target_idx] = curr_time

				target_idx += 1
				curr_pos = curr_target
				curr_target = np.asarray(path[target_idx])
				continue

			# Otherwise compute optimal direction of travel and move
			perp_target_vec = np.array((-target_vec[1], target_vec[0]))
			unit_perp_target_vec = perp_target_vec / np.linalg.norm(perp_target_vec)

			effective_vec = np.array((0.,0.))

			z = np.dot(flow_vec, unit_perp_target_vec)
			a = np.dot(unit_perp_target_vec, unit_perp_target_vec)
			b = 2*z*unit_perp_target_vec[1]
			c = z**2 - unit_perp_target_vec[0]**2 * boat_speed**2

			discriminant = b**2 - 4*a*c

			if discriminant < 0:
				print('Error: Not able to compute valid solution')
				print(f"discriminant: {discriminant}, z,a,b,c: {z},{a},{b},{c}")
				return None

			elif math.isclose(discriminant, 0., rel_tol=1e-5):
				# Double root, so just compute any solution
				y = -b / (2*a)
				x_squared = max(0., boat_speed**2 - y**2)
				x = np.sqrt(x_squared)

				boat_vecs = np.array([(x, y), (-x, y)])
				effective_vecs = boat_vecs + flow_vec
				unit_target_vec = target_vec / np.linalg.norm(target_vec)
				speed_vecs = np.dot(effective_vecs, unit_target_vec)
				max_speed_index = np.argmax(speed_vecs)
				if math.isclose(speed_vecs[max_speed_index], 0., rel_tol=1e-5):
					print('Error: Cannot make progress towards goal', effective_vecs[max_speed_index], target_vec)
					return None

				effective_vec = effective_vecs[max_speed_index]

			elif discriminant > 0:
				# Multiple solutions, choose solution resulting in greatest speed along desired path

				y1 = (-b + np.sqrt(discriminant)) / (2*a)
				y2 = (-b - np.sqrt(discriminant)) / (2*a)
				
				x1 = np.sqrt(boat_speed**2 - y1**2)
				x2 = np.sqrt(boat_speed**2 - y2**2)

				boat_vecs = np.array([(x1,y1), (-x1,y1), (x2,y2), (-x2,y2)])
				effective_vecs = boat_vecs + flow_vec
				unit_target_vec = target_vec / np.linalg.norm(target_vec)
				speed_vecs = np.dot(effective_vecs, unit_target_vec)
				max_speed_index = np.argmax(speed_vecs)
				if math.isclose(speed_vecs[max_speed_index], 0., rel_tol=1e-5):
					print('Error: Cannot make progress towards goal', effective_vecs[max_speed_index], target_vec)
					return None

				effective_vec = effective_vecs[max_speed_index]

			#print(f"Moving with effective vec: {effective_vec} with speed: {np.linalg.norm(effective_vec)}")

			curr_time += self._dt
			curr_pos += (effective_vec * self._dt)
			output.append((curr_time, curr_pos))

			if np.isnan(curr_pos[0]) or np.isnan(curr_pos[1]):
				print('nan coordinate', curr_pos, effective_vec, speed_vecs, max_speed_index)
				exit()

		return (output, wp_arrival_times)