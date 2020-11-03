import queue
import numpy as np

class AStar():

	def __init__(self, domain, heuristic, arrival_threshold=0.01, step_size=0.01):
		self._domain = domain
		self._heuristic = heuristic
		self._arrival_threshold = arrival_threshold
		self._step_size = step_size

	def plan_path(self, start, goal):
		connected_dirs = [(-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1)]
		search_dirs = [self._step_size*np.array(d) for d in connected_dirs]
		pq = queue.PriorityQueue()
		pq.put((0, start))

		came_from = {}
		cost_so_far = {}
		came_from[start] = None
		cost_so_far[start] = 0

		max_pq_size = 0
		while not pq.empty():
			print('pq: ',pq)
			max_pq_size = max(max_pq_size, pq.qsize())
			_, current_pos = pq.get()

			if self._heuristic.compute_cost(current_pos, goal) <= self._arrival_threshold:
				came_from[goal] = current_pos
				break

			for dir_step in search_dirs:
				next_pos = tuple(np.asarray(current_pos) + dir_step)
				if not self._domain.contains_point(next_pos):
					continue

				new_cost = cost_so_far[current_pos] + np.linalg.norm(dir_step)

				if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
					cost_so_far[next_pos] = new_cost
					priority = new_cost + self._heuristic.compute_cost(next_pos, goal)
					pq.put((priority, next_pos))
					came_from[next_pos] = current_pos

		# reconstruct path
		current_pos = goal
		path_coords = []
		print(start, goal, came_from)
		while current_pos != start:
			print(current_pos)
			path_coords.append(current_pos)
			current_pos = came_from[current_pos]
		path_coords.append(start)
		path_coords.reverse()

		return path_coords


class AStarPS(AStar):

	def plan_path(self, start, goal):
		astar_path = super().plan_path(start, goal)

		k = 0
		ps_path = [astar_path[k]]

		for i in range(1,len(astar_path)-1):
			if not self._domain.line_of_sight(ps_path[k], astar_path[i+1]):
				k += 1
				ps_path.append(astar_path[i])

		ps_path.append(goal)

		return ps_path
