import robot_primitives as rp
import sys
import json
import numpy as np
from pathlib import Path

from context import robot_utils as rut

scenario_name = sys.argv[1]
scenario_dir = Path(scenario_name)
domain_filename = str(scenario_dir / 'domain.json')
domain = rp.areas.Domain.from_file(domain_filename)

with (scenario_dir / 'field_params.json').open() as f:
	field_params = json.load(f)

field = None
field_type = field_params.pop('type')
if field_type == 'AsymmetricRadialChannelField':
	#field_params['undefined_value'] = (0., 0.)
	field = rp.fields.AsymmetricRadialChannelField(domain, **field_params)

dv = rut.plotting.DomainView(domain)

dv.plot_domain()
dv.plot_vf(field)
dv.hide_axes_labels()
dv.save(str(scenario_dir / 'scenario.png'))