import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from descartes.patch import PolygonPatch
import itertools

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

	def _draw(self):
		plt.axis('off')
		plt.show()
		plt.pause(self._pause_length)

	def clear_figure(self):
		self._fig.clf()
		self._ax = self._fig.add_subplot(1,1,1)
		self._ax.axis('equal')

	def center_view_to_domain(self):
		x_min, y_min, x_max, y_max = self._domain.bounds

		self._ax.set_xlim(x_min-1, x_max+1)
		self._ax.set_ylim(y_min-1, y_max+1)

		self._draw()

	def plot_domain(self, new_domain=None, domain_bg='xkcd:water blue', obstacle_bg='xkcd:reddish'):
		if new_domain is not None:
			self._domain = new_domain

		if self._domain is None:
			# No valid domain specified
			return

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

	def save(self, filename='default.png'):
		self._fig.savefig(filename, bbox_inches='tight', dpi=100)