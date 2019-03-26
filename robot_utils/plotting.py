import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from descartes.patch import PolygonPatch
import itertools
import shapely

class DomainView(object):
	""" A class for plotting various primitives on top of domains """

	def __init__(self, domain, title='Untitled', pause=0.00001):
		self._domain = domain
		x_min, y_min, x_max, y_max = domain.bounds
		x_dist = 1.0
		y_dist = (y_max - y_min)/x_max - x_min

		multiplier = 10

		plt.ion()
		self._fig = plt.figure(figsize=(x_dist*multiplier + 2,y_dist*multiplier), dpi=100) # todo make this use domain size
		self._ax = None
		self._clim = None

		self._pause_length = pause
		self._title = title

		self.plot_domain()

	def _draw(self):
		#plt.axis('off')
		plt.show()
		plt.pause(self._pause_length)

	def clear_figure(self):
		self._fig.clf()
		self._ax = self._fig.add_subplot(1,1,1)
		self._ax.axis('equal')
		self._ax.set_title(self._title)

	def center_view_to_domain(self):
		x_min, y_min, x_max, y_max = self._domain.bounds

		self._ax.set_xlim(x_min-1, x_max+1)
		self._ax.set_ylim(y_min-1, y_max+1)
		self._ax.axis('equal')

		self._draw()

	def change_domain(self, new_domain):
		self._domain = new_domain

		self.clear_figure()

	def plot_domain(self, domain_bg='xkcd:water blue', obstacle_bg='xkcd:reddish'):
		self.clear_figure()

		domain_patch = PolygonPatch(self._domain.polygon, facecolor=domain_bg, edgecolor=domain_bg, alpha=1.0, zorder=-10)
		self._ax.add_patch(domain_patch)

		for o in self._domain.obstacles.values():
			obstacle_patch = PolygonPatch(o.polygon, facecolor=obstacle_bg, edgecolor=obstacle_bg, alpha=1.0, zorder=-5)
			self._ax.add_patch(obstacle_patch)

		self.center_view_to_domain()

	def plot_velocity_contours(self, field, axis=0, show_contours=False):
		x_min, y_min, x_max, y_max = self._domain.bounds

		x_coords = np.linspace(x_min, x_max, 50)
		y_coords = np.linspace(y_min, y_max, 50)

		if axis < 2 and axis >= 0:
			wrapper = lambda x,y : field[(x,y)][axis]
		elif axis == 2:
			wrapper = lambda x,y : np.linalg.norm(np.array(field[(x,y)]))
		else:
			print('Specified axis is invalid:', axis)
			return

		X, Y = np.meshgrid(x_coords, y_coords)

		vec_wrapper = np.vectorize(wrapper)

		velocity = vec_wrapper(X, Y)

		if self._clim is None:
			self._clim = [velocity.min(), velocity.max()]

		self.clear_figure()

		if show_contours:
			contours = plt.contour(X, Y, velocity, 3, colors='black')
			plt.clabel(contours, inline=True, fontsize=8)

		img = self._ax.imshow(velocity, extent=[x_min, x_max, y_min, y_max], origin='lower',
										clim=self._clim, cmap='coolwarm', alpha=0.75)

		c = self._fig.colorbar(img, ax=self._ax)
		c.set_label('m/s')

		self.center_view_to_domain()

	def plot_vector_field(self, field):
		""" Quiver plot of vector field """
		x_min, y_min, x_max, y_max = self._domain.bounds
		x_dist = abs(x_max-x_min)
		y_dist = abs(y_max-y_min)

		x_coords = np.linspace(x_min, x_max, x_dist)
		y_coords = np.linspace(y_min, y_max, y_dist)

		wrapper = lambda x,y : field[(x,y)]
		vec_wrapper = np.vectorize(wrapper)

		X, Y = np.meshgrid(x_coords, y_coords)
		U, V = vec_wrapper(X, Y)
		magnitudes = np.sqrt(U**2 + V**2)

		if self._clim is None:
			self._clim = [magnitudes.min(), magnitudes.max()]

		self.clear_figure()

		quiver = self._ax.quiver(X, Y, U, V, magnitudes, clim=self._clim,
											angles='xy', scale_units='xy', scale=1, cmap='coolwarm')

		c = self._fig.colorbar(quiver, ax=self._ax)
		c.set_label('m/s')

		self.center_view_to_domain()

	def plot_configuration_space(self, vehicle_radius, domain_bg='xkcd:water blue', obstacle_bg='xkcd:reddish'):
		self.clear_figure()

		domain_patch = PolygonPatch(self._domain.polygon, facecolor=domain_bg, edgecolor=domain_bg, alpha=1.0, zorder=-10)
		self._ax.add_patch(domain_patch)

		offset_boundary, offset_obstacles = self._domain.get_configuration_space(vehicle_radius)

		domain_patch = PolygonPatch(offset_boundary, facecolor='xkcd:dark mint green', edgecolor='xkcd:dark mint green', alpha=1.0, zorder=-10)
		self._ax.add_patch(domain_patch)

		for o in offset_obstacles:
			obstacle_patch = PolygonPatch(o, facecolor=obstacle_bg, edgecolor=obstacle_bg, alpha=1.0, zorder=-5)
			self._ax.add_patch(obstacle_patch)

		for o in self._domain.obstacles.values():
			obstacle_patch = PolygonPatch(o.polygon, facecolor=obstacle_bg, edgecolor='xkcd:black', alpha=1.0, zorder=-5)
			self._ax.add_patch(obstacle_patch)

		self.center_view_to_domain()

	def plot_path(self, path, plot_endpoints=False):
		# need to check if path object is ok

		undefined_color = 'xkcd:steel grey'
		color_map = matplotlib.cm.get_cmap('Spectral')

		coord_pairs = zip(path.coord_list, path.coord_list[1:])
		segment_colors = []
		
		# Currently colors path by thrust constraint, change to be able to specify what to color by
		if path.is_constrained('thrust'):
			segment_colors.extend(map(lambda thrust_range: undefined_color if thrust_range is None else color_map(np.mean(thrust_range)), path.thrust[1:]))
		else:
			segment_colors.extend(itertools.repeat(undefined_color, path.size-1))

		print(path.coord_list)
		x,y = zip(*path.coord_list)
		self._ax.plot(x, y, 'o', color=undefined_color, markersize=4, zorder=1)
		if plot_endpoints:
			self._ax.plot(x[0], y[0], marker=5, color='xkcd:kiwi green', markersize=25)
			self._ax.plot(x[-1], y[-1], marker=9, color='xkcd:tomato', markersize=25)

		for seg_coords, seg_color in zip(coord_pairs, segment_colors):
			x,y = zip(*seg_coords)
			self._ax.plot(x, y, color=seg_color, linewidth=5, solid_capstyle='round', zorder=1)
		
		self.center_view_to_domain()

	def pretty_plot_path(self, path, offset=0.25, color='xkcd:black'):
		coord_pairs = zip(path.coord_list, path.coord_list[1:])

		plotted_segments = set()
		prev_cp = None

		for cp in coord_pairs:
			if prev_cp is not None:
				pcp = np.array(prev_cp)
				pcp_vec = pcp[1] - pcp[0]
				unit_vec = pcp_vec / np.linalg.norm(pcp_vec)
				offset_vec = unit_vec * offset
				segs = self._offset_overlapping_segment(cp, plotted_segments, offset_vec)
			else:
				segs = [cp]

			for s in segs:
				x,y = zip(*s)
				self._ax.plot(x,y, color=color, linewidth=2, solid_capstyle='round', zorder=1)

				plotted_segments.add(s)

			prev_cp = cp

		ingress = path.coord_list[0]
		egress = path.coord_list[-1]
		self._ax.plot(*ingress, marker=5, color='xkcd:kiwi green', markersize=25)
		self._ax.plot(*egress, marker="X", color='xkcd:tomato', markersize=25)

		self.center_view_to_domain()

	def _offset_overlapping_segment(self, segment, plotted_segments, offset):
		print(f"Call to offset func with params: segment={segment}, plotted_segments={plotted_segments}, and offset={offset}")
		slope_func = lambda p1, p2: (p2[1]-p1[1])/(p2[0]-p1[0])
		seg_start, seg_end = segment

		for seg in plotted_segments:
			s1, s2 = seg

			if s1[0] == s2[0]:
				# plotted seg is vertical line
				if seg_start[0] == seg_end[0] and seg_start[0] == s1[0]:
					# test segment is also vertical and colinear with plotted seg
					plotted_y_coords = [s1[1], s2[1]]
					plotted_y_coords.sort()

					segment_y_coords = [seg_start[1], seg_end[1]]
					segment_y_coords.sort()

					# check if test segment is bounded by plotted segment
					if plotted_y_coords[0] < segment_y_coords[0] and plotted_y_coords[1] > segment_y_coords[1]:
						start = np.array(seg_start)
						offset_start = tuple(start + offset)
						end = np.array(seg_end)
						offset_end = tuple(end + offset)

						new_seg = self._offset_overlapping_segment((offset_start, offset_end), plotted_segments, offset)

						segs = [(seg_start, offset_start), *new_seg, (offset_end, seg_end)]
						return segs

					# test to see if any either start or end of test segment lie between plotted segment coords
					#if (seg_start[1] > plotted_y_coords[0] and seg_start[1] < plotted_y_coords[1]) or (seg_end[1] > plotted_y_coords[0] and seg_end[1] < plotted_y_coords[1]):
					for y_coord in plotted_y_coords:
						if y_coord > segment_y_coords[0] and y_coord < segment_y_coords[1]:
							start = np.array(seg_start)
							offset_start = tuple(start + offset)
							end = np.array(seg_end)
							offset_end = tuple(end + offset)

							new_seg = self._offset_overlapping_segment((offset_start, offset_end), plotted_segments, offset)

							segs = [(seg_start, offset_start), *new_seg, (offset_end, seg_end)]
							return segs

			else:
				# plotted seg is not vertical line
				slope = slope_func(s1, s2)
				eq = lambda x: slope*(x-s1[0])+s1[1]

				if seg_start[1] == eq(seg_start[0]) and seg_end[1] == eq(seg_end[0]):
					# test segment is colinear to plotted seg
					plotted_x_coords = [s1[0], s2[0]]
					plotted_x_coords.sort()

					segment_x_coords = [seg_start[0], seg_end[0]]
					segment_x_coords.sort()

					if plotted_x_coords[0] < segment_x_coords[0] and plotted_x_coords[1] > segment_x_coords[1]:
						start = np.array(seg_start)
						offset_start = tuple(start + offset)
						end = np.array(seg_end)
						offset_end = tuple(end + offset)

						new_seg = self._offset_overlapping_segment((offset_start, offset_end), plotted_segments, offset)

						segs = [(seg_start, offset_start), *new_seg, (offset_end, seg_end)]
						return segs

					print(seg_start, plotted_x_coords, seg_end)
					for x_coord in plotted_x_coords:
						if x_coord > segment_x_coords[0] and x_coord < segment_x_coords[1]:
							start = np.array(seg_start)
							offset_start = tuple(start + offset)
							end = np.array(seg_end)
							offset_end = tuple(end + offset)

							new_seg = self._offset_overlapping_segment((offset_start, offset_end), plotted_segments, offset)

							segs = [(seg_start, offset_start), *new_seg, (offset_end, seg_end)]
							return segs

		return [segment]

	def save(self, filename='default.png'):
		self._fig.savefig(filename, bbox_inches='tight', dpi=100)

	@property
	def title(self):
		return self._title
	
	@title.setter
	def title(self, new_title):
		self._title = new_title
		self._ax.set_title(new_title)