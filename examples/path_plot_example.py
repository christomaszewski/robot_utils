import robot_primitives as rp

from context import robot_utils as rut

domain_width = 6.

domain = rp.areas.Domain.from_box_corners((0,0), (domain_width, domain_width))

dv = rut.plotting.DomainView(domain)
dv.plot_domain()

# Boustrphedon with overlapping segments
path_coords = [(1,1), (1,5), (4,5), (4,1), (2,1), (2,5), (5,5), (5,1), (3,1), (3,5)]

# Spiral with overlapping vertices
path_coords = [(1,1), (1,5), (5,5), (5,1), (1,1), (2,2), (2,4), (4,4), (4,2), (2,2), (3,3)]

path = rp.paths.ConstrainedPath(path_coords)

dv.pretty_plot_path(path, offset=0.25)
dv.save('test_path_plot.png')