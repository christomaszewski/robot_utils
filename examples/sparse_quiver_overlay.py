import robot_primitives as rp

from context import robot_utils as rut

domain_width = 10.
domain_height= 10.
max_flow_speed = -2.

domain = rp.areas.Domain.from_box_corners((0,0), (domain_width, domain_height))
flow_field = rp.fields.VectorField.from_channel_flow_model(channel_width=domain_width, max_velocity=max_flow_speed)

dv = rut.plotting.DomainView(domain, title='', domain_buffer=0.)
dv.plot_domain()
#dv.plot_vector_field(flow_field, cell_size=(1.,2), vec_pos=(0.5,2.))
dv.plot_vector_field(flow_field, cell_size=(1.,4))
dv.toggle_axes()
dv.save('quiver_overlay.png')
