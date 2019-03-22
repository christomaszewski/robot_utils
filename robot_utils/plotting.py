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
		self._fig = plt.figure(figsize=(x_dist*multiplier,y_dist*multiplier), dpi=100) # todo make this use domain size
		self._ax = None

		self._pause_length = pause
		self._title = title

		self.plot_domain()

	def _draw(self):
		plt.axis('off')
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

		x_min, y_min, x_max, y_max = self._domain.bounds

		self._ax.set_xlim(x_min-1, x_max+1)
		self._ax.set_ylim(y_min-1, y_max+1)

		self._draw()

	def plot_velocity_contours(self, field, axis=0):
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

		self.clear_figure()

		contours = plt.contour(X, Y, velocity, 3, colors='black')
		plt.clabel(contours, inline=True, fontsize=8)

		plt.imshow(velocity, extent=[x_min, x_max, y_min, y_max], origin='lower',
           cmap='Spectral', alpha=0.5)


		self.center_view_to_domain()

	def plot_path(self, path, plot_endpoints=False):
		# need to check if path object is ok

		undefined_color = 'xkcd:silver'
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
		
		self._draw()

	def pretty_plot_path(self, path, x_offset=0., y_offset=0.):
		color = 'xkcd:silver'
		coord_pairs = zip(path.coord_list, path.coord_list[1:])

		plotted_segments = set()

		for cp in coord_pairs:
			seg = shapely.geometry.LineString(cp)

			seg = self._offset_overlapping_segment(seg, plotted_segments, x_offset, y_offset)

			x,y = zip(*seg.coords)
			self._ax.plot(x,y, color=color, linewidth=3, solid_capstyle='round', zorder=1)

			plotted_segments.add(seg)

		self._draw()


	def _offset_overlapping_segment(self, segment, plotted_segments, x_offset, y_offset):
		for seg in plotted_segments:
			if segment.intersects(seg):
				new_coords = [(x+x_offset, y+y_offset) for coord in segment.coords for x,y in coord]
				segment.coords = new_coords

				segment = self._offset_overlapping_segment(segment, plotted_segments, x_offset, y_offset)

		return segment

	def save(self, filename='default.png'):
		self._fig.savefig(filename, bbox_inches='tight', dpi=100)

	@property
	def title(self):
		return self._title
	
	@title.setter
	def title(self, new_title):
		self._title = new_title