import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from descartes.patch import PolygonPatch
import itertools
import shapely
import cartopy
import cartopy.crs as ccrs
from cartopy.io.img_tiles import MapboxTiles, GoogleTiles

class MapView(object):
	""" A class for plotting various primitives on top of a map """

	api_key = 'pk.eyJ1IjoiY2hyaXN0b21hc3pld3NraSIsImEiOiJjanJtN2h1OTAwZ2lnM3ltdDBmZDFjc3FyIn0.2i83Ad3s4mi9DR6ZLG-CFg'

	def __init__(self, bounds, title='', utm_zone=17, pause=0.00001, extent_buffer=10., map_source='mapbox', map_style='satellite', z_level=17):
		self._bounds = bounds
		x_min, y_min, x_max, y_max = bounds

		plt.ion()
		self._fig = plt.figure(figsize=(12,10), dpi=100) # todo make this use domain size
		self._ax = None
		self._clim = None
		self._z_level = z_level

		self._pause_length = pause
		self._title = title
		self._show_axes = True

		self._utm_zone = utm_zone

		if map_source.lower() == 'google':
			self._imagery = GoogleTiles(style=map_style.lower())
		else:
			map_style = 'terrain-rgb'
			self._imagery = MapboxTiles(MapView.api_key, map_style)
			self._imagery._image_url = self._image_url

		self._ax = self._fig.add_subplot(1,1,1, projection=self._imagery.crs)
		self._ax.axis('equal')
		
		self._extents = [x_min-extent_buffer, x_max+extent_buffer, y_min-extent_buffer, y_max+extent_buffer]

		self._ax.set_extent(self._extents, crs=ccrs.UTM(self._utm_zone))

		self._ax.add_image(self._imagery, self._z_level) # Satellite 17
		
	@classmethod
	def from_domain(cls, domain, **kwargs):
		return cls(domain.bounds, **kwargs)

	def _image_url(self, tile):
		x, y, z = tile
		# /styles/v1/{username}/{style_id}/tiles/{tilesize}/{z}/{x}/{y}{@2x}
		#url = (f"https://api.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}@2x.png?access_token={MapView.api_key}")
		url = (f"https://api.mapbox.com/styles/v1/christomaszewski/ck72d2m0o0igw1io7fibjekq9/tiles/256/{z}/{x}/{y}@2x?access_token={MapView.api_key}")
		return url

	def _draw(self):
		#plt.axis('off')
		plt.show()
		plt.pause(self._pause_length)

	def clear_figure(self):
		self._fig.clf()
		self._ax = self._fig.add_subplot(1,1,1, projection=self._imagery.crs)
		self._ax.axis('equal')
		self._ax.set_title(self._title)
		self._ax.set_extent(self._extents, crs=ccrs.UTM(self._utm_zone))

		self._ax.add_image(self._imagery, self._z_level)

	def plot_domain_boundary(self, domain, color='xkcd:water blue'):
		x,y = domain.polygon.exterior.xy
		self._ax.plot(x,y, color=color, linewidth=3, solid_capstyle='round', zorder=1, transform=ccrs.UTM(self._utm_zone))

		self._draw()

	def plot_vf(self, field, num_cells=(25,25), scale=0.05, pivot='mid', minshaft=1.5, ticks=None, clim=None):
		x_min, y_min, x_max, y_max = self._domain.bounds

		x_cell_count = num_cells[0]
		y_cell_count = num_cells[1]

		wrapper = lambda x,y: field[(x,y)]
		vectorized_func = np.vectorize(wrapper)

		x_grid, y_grid = np.mgrid[x_min:x_max:(x_cell_count*1j),y_min:y_max:(y_cell_count*1j)] 

		x_samples, y_samples = vectorized_func(x_grid, y_grid)

		magnitudes = np.sqrt(x_samples**2 + y_samples**2)

		if not clim:
			clim = [np.nanmin(magnitudes), np.nanmax(magnitudes)]

		print("plotting quiver")
		q = self._ax.quiver(x_grid, y_grid, x_samples, y_samples, magnitudes, 
						clim=clim, angles='xy', scale_units='xy', scale=scale, pivot=pivot, minshaft=minshaft, 
						cmap=plt.get_cmap('rainbow'), transform=ccrs.UTM(self._utm_zone))

		self._draw()

		if not ticks:
			ticks = [float(i) for i in np.linspace(*clim, 6)]

		cax = self._fig.add_axes([self._ax.get_position().x1+0.01,self._ax.get_position().y0,0.02,self._ax.get_position().height])
		c = self._fig.colorbar(q, cax=cax, ticks=ticks)
		c.set_label('Flow Speed (m/s)', size=16, labelpad=10)
		c.ax.tick_params(labelsize=16)

		self._draw()

	def plot_vector_field(self, field):
		x_min, y_min, x_max, y_max = self._bounds

		x_cell_count = 25
		y_cell_count = 25

		wrapper = lambda x,y: field[(x,y)]
		vectorized_func = np.vectorize(wrapper)

		x_grid, y_grid = np.mgrid[x_min:x_max:(x_cell_count*1j),y_min:y_max:(y_cell_count*1j)] 

		x_samples, y_samples = vectorized_func(x_grid, y_grid)

		magnitudes = np.sqrt(x_samples**2 + y_samples**2)

		clim = [np.nanmin(magnitudes), np.nanmax(magnitudes)]

		print("plotting quiver")
		q = self._ax.quiver(x_grid, y_grid, x_samples, y_samples, magnitudes, transform=ccrs.UTM(self._utm_zone), 
						clim=clim, angles='xy', scale_units='xy', scale=0.05, pivot='mid', #minshaft=2.0, 
						cmap=plt.get_cmap('rainbow'))

		c = self._fig.colorbar(q, ax=self._ax)
		c.set_label('m/s')

		self._draw()

	def plot_segments(self, segments, line_color='xkcd:teal blue', line_width=3, marker_size=5, marker_color='xkcd:teal blue'):
		for idx, c_coords in enumerate(segments):

			crs = ccrs.UTM(self._utm_zone)


			x,y = zip(*c_coords)
			self._ax.plot(x, y, 'o', color=marker_color, markersize=marker_size, zorder=1, transform=crs)
			self._ax.plot(x, y, color=line_color, linewidth=line_width, solid_capstyle='round', transform=crs)

			self._draw()


	def plot_constraints(self, constraints, color='xkcd:teal blue', plot_direction=True, plot_sequence=True, line_width=3):
		for idx, c in enumerate(constraints):
			c_coords = c.get_coord_list()

			crs = ccrs.UTM(self._utm_zone)

			x,y = zip(*c_coords)
			self._ax.plot(x, y, 'o', color=color, markersize=4, zorder=1, transform=crs)
			self._ax.plot(x, y, color=color, linewidth=line_width, solid_capstyle='round', transform=crs)

			if c.is_constrained('direction') and plot_direction:
				#direction = c.direction
				coord_pairs = zip(c_coords, c_coords[1:])
				for cp in coord_pairs:
					pts = [np.array(pt) for pt in cp]
					mid_pt = np.mean(pts, axis=0)
					# Don't need to do any modification for direction because get_coord_list() should return the
					# coord list in order if direction is constrained
					#seg_vec = pts[direction[1]] - pts[direction[0]]
					seg_vec = pts[1] - pts[0]
					seg_len = np.linalg.norm(seg_vec)
					seg_dir = seg_vec/seg_len
					arrow_len = 0.1 * seg_len

					arrow_base = mid_pt - 0.5*arrow_len*seg_dir
					arrow_vec = arrow_len*seg_dir
					#print(arrow_base[0], arrow_base[1], arrow_vec[0], arrow_vec[1])
					self._ax.arrow(arrow_base[0], arrow_base[1], arrow_vec[0], arrow_vec[1], fc=color,
										shape='full', lw=0, length_includes_head=True, head_width=arrow_len/2., zorder=2, transform=crs)


			if plot_sequence:
				self._ax.annotate(f"{idx}",
										xy=c_coords[0], xycoords='data',
										xytext=(0, -70), textcoords='offset points',
										size=20,
										bbox=dict(boxstyle="round",
										fc=(1.0, 0.7, 0.7),
										ec=(1., .5, .5)),
										arrowprops=dict(arrowstyle="wedge,tail_width=1.",
										fc=(1.0, 0.7, 0.7), ec=(1., .5, .5),
										patchA=None,
										patchB=None,
										relpos=(0.2, 0.8),
										connectionstyle="arc3,rad=-0.1"))

			self._draw()

	def plot_plan(self, path, ingress_color='xkcd:emerald green', egress_color='xkcd:tomato', path_color='xkcd:teal blue', marker_color='xkcd:black', path_width=3):
		x,y = zip(*path.coord_list)

		crs = ccrs.UTM(self._utm_zone)

		self._ax.plot(x[:2], y[:2], color=ingress_color, linewidth=path_width, solid_capstyle='round', zorder=2, transform=crs)
		self._ax.plot(x[1:-1], y[1:-1], color=path_color, linewidth=path_width, solid_capstyle='round', zorder=3, transform=crs)
		self._ax.plot(x[-2:], y[-2:], color=egress_color, linewidth=path_width, solid_capstyle='round', zorder=4, transform=crs)

		self._ax.plot(x[0], y[0], 'o', color=marker_color, markersize=path_width+2, zorder=5, transform=ccrs.UTM(self._utm_zone))

		self._draw()

	def plot_path(self, path, color='xkcd:steel grey', plot_points=False, path_width=2):
		# need to check if path object is ok

		undefined_color = color
		color_map = matplotlib.cm.get_cmap('Spectral')

		coord_pairs = zip(path.coord_list, path.coord_list[1:])
		segment_colors = []
		
		# Currently colors path by thrust constraint, change to be able to specify what to color by
		if path.is_constrained('thrust'):
			segment_colors.extend(map(lambda thrust_range: undefined_color if thrust_range is None else color_map(np.mean(thrust_range)), path.thrust[1:]))
		else:
			segment_colors.extend(itertools.repeat(undefined_color, path.size-1))

		x,y = zip(*path.coord_list)

		if plot_points:
			self._ax.plot(x, y, 'o', color=undefined_color, markersize=4, zorder=1, transform=ccrs.UTM(self._utm_zone))
		
		for seg_coords, seg_color in zip(coord_pairs, segment_colors):
			x,y = zip(*seg_coords)
			self._ax.plot(x, y, color=seg_color, linewidth=path_width, solid_capstyle='round', zorder=1, transform=ccrs.UTM(self._utm_zone))

		self._draw()

	# Legacy plot_path func
	"""
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
		self._ax.plot(x, y, 'o', color=undefined_color, markersize=4, zorder=1, transform=ccrs.UTM(17))
		if plot_endpoints:
			self._ax.plot(x[0], y[0], marker=5, color='xkcd:kiwi green', markersize=25, transform=ccrs.UTM(17))
			self._ax.plot(x[-1], y[-1], marker=9, color='xkcd:tomato', markersize=25, transform=ccrs.UTM(17))

		for seg_coords, seg_color in zip(coord_pairs, segment_colors):
			x,y = zip(*seg_coords)
			self._ax.plot(x, y, color=seg_color, linewidth=5, solid_capstyle='round', zorder=1, transform=ccrs.UTM(17))
		
		self._draw()
	"""

	def pretty_plot_path(self, path, offset=0.25, color='xkcd:steel grey', linewidth=4):
		coord_pairs = zip(path.coord_list, path.coord_list[1:])

		plotted_segments = set()
		plotted_verts = set()
		prev_cp = None
		prev_seg_adjusted = False

		for cp in coord_pairs:
			if prev_seg_adjusted:
				# Need to adjust start point of current segment to match offset end point of prev segment
				cp = (prev_cp[-1], cp[1])
				prev_seg_adjusted = False

			if prev_cp is not None:
				pcp = np.array(prev_cp)
				pcp_vec = pcp[1] - pcp[0]
				unit_vec = pcp_vec / np.linalg.norm(pcp_vec)
				offset_vec = unit_vec * offset
				segs = self._offset_overlapping_segment(cp, plotted_segments, offset_vec)
			else:
				segs = [cp]

			for s in segs[:-1]:
				x,y = zip(*s)
				self._ax.plot(x,y, color=color, linewidth=linewidth, solid_capstyle='round', zorder=1, transform=ccrs.UTM(self._utm_zone))

				plotted_segments.add(s)
				plotted_verts.update(s)

			last_seg = segs[-1]
			if last_seg[-1] in plotted_verts:
				# shorten last segment by offset distance
				s = np.array(last_seg)
				s_vec = s[0] - s[1]
				unit_s_vec = s_vec / np.linalg.norm(s_vec)
				offset_vec = unit_s_vec * offset
				new_end_point = tuple(s[1] + offset_vec)
				last_seg = (last_seg[0], new_end_point)
				prev_seg_adjusted = True

			x,y = zip(*last_seg)
			self._ax.plot(x,y, color=color, linewidth=linewidth, solid_capstyle='round', zorder=1, transform=ccrs.UTM(self._utm_zone))
			plotted_segments.add(last_seg)
			plotted_verts.update(last_seg)

			prev_cp = last_seg

		ingress = path.coord_list[0]
		egress = prev_cp[-1]
		self._ax.plot(*ingress, marker=5, color='xkcd:kiwi green', markersize=25, transform=ccrs.UTM(self._utm_zone))
		self._ax.plot(*egress, marker="X", color='xkcd:tomato', markersize=25, transform=ccrs.UTM(self._utm_zone))

		self._draw()

	def _offset_overlapping_segment(self, segment, plotted_segments, offset):
		""" Does not work for arbitary domain orientations!!! """
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

	def save(self, filename):
		self._fig.savefig(filename, bbox_inches='tight', dpi=100)


class DomainView(object):
	""" A class for plotting various primitives on top of domains """

	def __init__(self, domain, title='', pause=0.00001, domain_buffer=2.5):
		self._domain = domain
		x_min, y_min, x_max, y_max = domain.bounds
		x_dist = 20.
		y_dist = 20.*(y_max - y_min)/(x_max - x_min)

		plt.ion()

		self._fig = plt.figure(figsize=(x_dist, y_dist), dpi=100) # todo make this use domain size
		self._ax = self._fig.add_subplot(1,1,1)
		self._ax.set_aspect('equal')
		self._clim = None

		self._domain_buffer = domain_buffer
		self._pause_length = pause
		self._title = title
		self._show_axes = True

		self.plot_domain()

	def _draw(self, pause=None):
		if pause is None:
			pause = self._pause_length

		plt.show()
		plt.pause(pause)

	def clear_figure(self):
		self._fig.clf()
		self._ax = self._fig.add_subplot(1,1,1)
		self._ax.set_aspect('equal')
		#self._ax.axis('equal')
		self._ax.set_title(self._title)

	def toggle_axes(self, show_axes=None):
		self._show_axes = not self._show_axes

		# Allow for explicit override using suppled func arg
		if show_axes:
			self._show_axes = show_axes

		if self._show_axes:
			self._ax.axis('on')
		else:
			self._ax.axis('off')

	def center_view_to_domain(self):
		x_min, y_min, x_max, y_max = self._domain.bounds

		self._ax.set_xlim(x_min-self._domain_buffer, x_max+self._domain_buffer)
		self._ax.set_ylim(y_min-self._domain_buffer, y_max+self._domain_buffer)
		#self._ax.axis('equal')
		#self._ax.set_aspect('equal', 'box')
		self._draw()

	def change_domain(self, new_domain):
		self._domain = new_domain

		self.clear_figure()

	def plot_domain_boundary(self, color='xkcd:water blue'):
		x,y = self._domain.polygon.exterior.xy
		self._ax.plot(x,y, color=color, linewidth=3, solid_capstyle='round', zorder=2)

		self.center_view_to_domain()

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

		#self.clear_figure()

		if show_contours:
			contours = self._ax.contour(X, Y, velocity, 3, colors='black')
			self._ax.clabel(contours, inline=True, fontsize=8)

		img = self._ax.imshow(velocity, extent=[x_min, x_max, y_min, y_max], origin='lower',
										clim=self._clim, cmap='coolwarm', alpha=0.75)

		c = self._fig.colorbar(img, ax=self._ax)
		c.set_label('m/s')

		self.center_view_to_domain()

	def plot_vf(self, field, num_cells=(25,25), scale=0.05, pivot='mid', minshaft=1.5, ticks=None, clim=None, label_size=18, **kwargs):
		x_min, y_min, x_max, y_max = self._domain.bounds

		x_cell_count = num_cells[0]
		y_cell_count = num_cells[1]

		wrapper = lambda x,y: field[(x,y)]
		vectorized_func = np.vectorize(wrapper)

		x_grid, y_grid = np.mgrid[x_min:x_max:(x_cell_count*1j),y_min:y_max:(y_cell_count*1j)] 

		x_samples, y_samples = vectorized_func(x_grid, y_grid)

		magnitudes = np.sqrt(x_samples**2 + y_samples**2)

		if not clim:
			clim = [np.nanmin(magnitudes), np.nanmax(magnitudes)]

		print("plotting quiver")
		q = self._ax.quiver(x_grid, y_grid, x_samples, y_samples, magnitudes, 
						clim=clim, angles='xy', scale_units='xy', scale=scale, pivot=pivot, minshaft=minshaft, 
						cmap=plt.get_cmap('rainbow'), **kwargs)

		self._draw()

		if not ticks:
			ticks = [float(i) for i in np.linspace(*clim, 6)]

		cax = self._fig.add_axes([self._ax.get_position().x1+0.01,self._ax.get_position().y0,0.02,self._ax.get_position().height])
		c = self._fig.colorbar(q, cax=cax, ticks=ticks)
		c.set_label('Flow Speed (m/s)', size=label_size, labelpad=10)
		c.ax.tick_params(labelsize=label_size)

		self._draw()

	def plot_vector_field(self, field, cell_size=(1.,1.), vec_pos=None):
		""" Quiver plot of vector field """
		if not vec_pos:
			vec_pos = (cell_size[0]/2, cell_size[1]/2)

		x_min, y_min, x_max, y_max = self._domain.bounds
		x_dist = abs(x_max-x_min)
		y_dist = abs(y_max-y_min)

		x_num_cells = np.floor(x_dist / cell_size[0])
		y_num_cells = np.floor(y_dist / cell_size[1])
		#x_coords = np.linspace(x_min+0.5, x_max-0.5, x_dist-1)
		#y_coords = np.linspace(y_min+2., y_max-0.25, y_dist/2)
		#x_coords = np.linspace(x_min+cell_size[0]/2, x_max-cell_size[0]/2, x_num_cells-1)
		#y_coords = np.linspace(y_min+cell_size[1]/2, y_max-cell_size[1]/2, y_num_cells-1)
		x_coords = np.linspace(x_min+vec_pos[0], x_max-cell_size[0]+vec_pos[0], x_num_cells)
		y_coords = np.linspace(y_min+vec_pos[1], y_max-cell_size[1]+vec_pos[1], y_num_cells)

		wrapper = lambda x,y : field[(x,y)]
		vec_wrapper = np.vectorize(wrapper)

		X, Y = np.meshgrid(x_coords, y_coords)
		U, V = vec_wrapper(X, Y)
		magnitudes = np.sqrt(U**2 + V**2)

		if self._clim is None:
			self._clim = [magnitudes.min(), magnitudes.max()]

		#self.clear_figure()

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

	def plot_offsets(self, vehicle_radius, sensor_radius, domain_bg='xkcd:water blue', config_bg='xkcd:dark mint green', sensor_bg='xkcd:marigold'):
		self.clear_figure()

		domain_patch = PolygonPatch(self._domain.polygon, facecolor=domain_bg, edgecolor=domain_bg, alpha=1.0, zorder=-10)
		self._ax.add_patch(domain_patch)

		config_boundary, _ = self._domain.get_configuration_space(vehicle_radius)

		config_patch = PolygonPatch(config_boundary, facecolor=config_bg, edgecolor=config_bg, alpha=1.0, zorder=-5)
		self._ax.add_patch(config_patch)

		sensor_boundary, _ = self._domain.get_configuration_space(sensor_radius)

		sensor_patch = PolygonPatch(sensor_boundary, facecolor=sensor_bg, edgecolor=sensor_bg, alpha=1.0, zorder=-3)
		self._ax.add_patch(sensor_patch)

		self.center_view_to_domain()

	def plot_path(self, path, color='xkcd:steel grey', plot_points=False, path_width=2):
		# need to check if path object is ok

		undefined_color = color
		color_map = matplotlib.cm.get_cmap('Spectral')

		coord_pairs = zip(path.coord_list, path.coord_list[1:])
		segment_colors = []
		
		# Currently colors path by thrust constraint, change to be able to specify what to color by
		if path.is_constrained('thrust'):
			segment_colors.extend(map(lambda thrust_range: undefined_color if thrust_range is None else color_map(np.mean(thrust_range)), path.thrust[1:]))
		else:
			segment_colors.extend(itertools.repeat(undefined_color, path.size-1))

		x,y = zip(*path.coord_list)

		if plot_points:
			self._ax.plot(x, y, 'o', color=undefined_color, markersize=4, zorder=1)
		
		for seg_coords, seg_color in zip(coord_pairs, segment_colors):
			x,y = zip(*seg_coords)
			self._ax.plot(x, y, color=seg_color, linewidth=path_width, solid_capstyle='round', zorder=1)
		
		self.center_view_to_domain()

	def hide_axes_labels(self):
		self._ax.set_xticks([])
		self._ax.set_yticks([])

	def animate_path(self, path, color='xkcd:steel grey', pause=0.2):

		coord_pairs = zip(path.coord_list, path.coord_list[1:])

		for seg_coords in zip(path.coord_list, path.coord_list[1:]):
			x,y = zip(*seg_coords)
			self._ax.plot(x, y, color=color, linewidth=2, solid_capstyle='round', zorder=1)
			self._draw(pause)
		
		self.center_view_to_domain()

	def plot_constraints(self, constraints, color='xkcd:melon', plot_direction=True, plot_sequence=True, line_width=2):
		for idx, c in enumerate(constraints):
			c_coords = c.get_coord_list()

			x,y = zip(*c_coords)
			self._ax.plot(x, y, 'o', color=color, markersize=4, zorder=1)
			self._ax.plot(x, y, color=color, linewidth=line_width, solid_capstyle='round')

			if c.is_constrained('direction') and plot_direction:
				#direction = c.direction
				coord_pairs = zip(c_coords, c_coords[1:])
				for cp in coord_pairs:
					pts = [np.array(pt) for pt in cp]
					mid_pt = np.mean(pts, axis=0)
					# Don't need to do any modification for direction because get_coord_list() should return the
					# coord list in order if direction is constrained
					#seg_vec = pts[direction[1]] - pts[direction[0]]
					seg_vec = pts[1] - pts[0]
					seg_len = np.linalg.norm(seg_vec)
					seg_dir = seg_vec/seg_len
					arrow_len = 0.1 * seg_len

					arrow_base = mid_pt - 0.5*arrow_len*seg_dir
					arrow_vec = arrow_len*seg_dir
					#print(arrow_base[0], arrow_base[1], arrow_vec[0], arrow_vec[1])
					self._ax.arrow(arrow_base[0], arrow_base[1], arrow_vec[0], arrow_vec[1], fc=color,
										shape='full', lw=0, length_includes_head=True, head_width=arrow_len/2., zorder=2)


			if plot_sequence:
				self._ax.annotate(f"{idx}",
										xy=c_coords[0], xycoords='data',
										xytext=(0, -70), textcoords='offset points',
										size=20,
										bbox=dict(boxstyle="round",
										fc=(1.0, 0.7, 0.7),
										ec=(1., .5, .5)),
										arrowprops=dict(arrowstyle="wedge,tail_width=1.",
										fc=(1.0, 0.7, 0.7), ec=(1., .5, .5),
										patchA=None,
										patchB=None,
										relpos=(0.2, 0.8),
										connectionstyle="arc3,rad=-0.1"))

			self._draw()

	def plot_coverage(self, coord_list, sensor_radius):
		path_line = shapely.geometry.LineString(coord_list)
		sensor_coverage = path_line.buffer(sensor_radius)

		coverage_patch = PolygonPatch(sensor_coverage, facecolor='xkcd:goldenrod', edgecolor='xkcd:dark grey', alpha=0.5, zorder=1)
		self._ax.add_patch(coverage_patch)

		self.center_view_to_domain()

	def plot_robot(self, coords, vehicle_radius, sensor_radius, robot_color='xkcd:grey', sensor_color='xkcd:goldenrod'):
		pt = shapely.geometry.Point(*coords)

		vehicle_patch = PolygonPatch(pt.buffer(vehicle_radius), facecolor=robot_color, edgecolor='xkcd:dark grey', alpha=1.0, zorder=2)
		sensor_patch = PolygonPatch(pt.buffer(sensor_radius), facecolor=sensor_color, edgecolor='xkcd:dark grey', alpha=1.0, zorder=1)

		self._ax.add_patch(sensor_patch)
		self._ax.add_patch(vehicle_patch)

		self.center_view_to_domain()

	def plot_endpoints(self, path, ingress_marker=5, ingress_color='xkcd:kiwi green', egress_marker='X', egress_color='xkcd:tomato', marker_size=25):
		ingress = path.coord_list[0]
		egress = path.coord_list[-1]
		self._ax.plot(*ingress, marker=ingress_marker, color=ingress_color, markersize=marker_size)
		self._ax.plot(*egress, marker=egress_marker, color=egress_color, markersize=marker_size)

		self._draw()

	def pretty_plot_path(self, path, offset=0.25, color='xkcd:black', linewidth=2):
		tuple_coords = list(map(tuple, path.coord_list))
		coord_pairs = zip(tuple_coords, tuple_coords[1:])

		plotted_segments = set()
		plotted_verts = set()
		prev_cp = None
		prev_seg_adjusted = False

		for cp in coord_pairs:
			if prev_seg_adjusted:
				# Need to adjust start point of current segment to match offset end point of prev segment
				cp = (prev_cp[-1], cp[1])
				prev_seg_adjusted = False

			if prev_cp is not None:
				pcp = np.array(prev_cp)
				pcp_vec = pcp[1] - pcp[0]
				unit_vec = pcp_vec / np.linalg.norm(pcp_vec)
				offset_vec = unit_vec * offset
				segs = self._offset_overlapping_segment(cp, plotted_segments, offset_vec)
			else:
				segs = [cp]

			for s in segs[:-1]:
				x,y = zip(*s)
				self._ax.plot(x,y, color=color, linewidth=linewidth, solid_capstyle='round', zorder=1)

				plotted_segments.add(s)
				plotted_verts.update(s)

			last_seg = segs[-1]
			if last_seg[-1] in plotted_verts:
				# shorten last segment by offset distance
				s = np.array(last_seg)
				s_vec = s[0] - s[1]
				unit_s_vec = s_vec / np.linalg.norm(s_vec)
				offset_vec = unit_s_vec * offset
				new_end_point = tuple(s[1] + offset_vec)
				last_seg = (last_seg[0], new_end_point)
				prev_seg_adjusted = True

			x,y = zip(*last_seg)
			self._ax.plot(x,y, color=color, linewidth=linewidth, solid_capstyle='round', zorder=1)

			plotted_segments.add(last_seg)
			plotted_verts.update(last_seg)

			prev_cp = last_seg

		ingress = path.coord_list[0]
		egress = prev_cp[-1]
		self._ax.plot(*ingress, marker=5, color='xkcd:kiwi green', markersize=25)
		self._ax.plot(*egress, marker="X", color='xkcd:tomato', markersize=25)

		self.center_view_to_domain()

	def _offset_overlapping_segment(self, segment, plotted_segments, offset):
		print(f"Call to offset func with params: segment={segment}, and offset={offset}")
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

					#print(seg_start, plotted_x_coords, seg_end)
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