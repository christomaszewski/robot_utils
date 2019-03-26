import robot_primitives as rp

from context import robot_utils as rut

domain_width = 30.
max_flow_speed = -2.
pylon_width = 5.

domain = rp.areas.Domain.from_box_corners((0,0), (domain_width, domain_width))
#flow_field = rp.fields.VectorField.from_channel_flow_model(channel_width=domain_width, max_velocity=max_flow_speed)
flow_field = rp.fields.VectorField.from_channel_flow_with_pylon(channel_width=domain_width, max_velocity=max_flow_speed, pylon_bounds=(domain_width/2. - pylon_width/2., domain_width/2. + pylon_width/2.))

dv = rut.plotting.DomainView(domain, title='Channel Flow Downstream of Pylon')

dv.plot_velocity_contours(flow_field, axis=2, show_contours=True)
dv.save('velocity_contours.png')

dv.plot_vector_field(flow_field)
dv.save('vector_field.png')
