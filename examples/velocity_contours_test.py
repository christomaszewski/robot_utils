import robot_primitives as rp

from context import robot_utils as rut

domain_width = 20.
max_flow_speed = -2.

domain = rp.areas.Domain.from_box_corners((0,0), (domain_width, domain_width))
#flow_field = rp.fields.VectorField.from_channel_flow_model(channel_width=domain_width, max_velocity=max_flow_speed)
flow_field = rp.fields.VectorField.from_channel_flow_with_pylon(channel_width=domain_width, max_velocity=max_flow_speed, pylon_bounds=(8,12))

dv = rut.plotting.DomainView(domain)

dv.plot_velocity_contours(flow_field, axis=1)

dv.save('contours.png')