"""
Helper for importing package level settings from the user's home directory

Allows a particular package (eg `awrams.visualisation`) to import user specific
settings from either `~/.awrams/visualisation.py` or `~/.awrams/settings.py`
using get_settings('visualisation')

Intended use is from a package specific `settings.py` of equivalent that looks like

SOME_SETTING=#default value
SOME_OTHER_SETTING=#default value

# end of file, load and override defaults with anything from the user's
# own settings file
from .settings_manager import get_settings as _get_settings
exec(_get_settings('package_name'))
"""

import os

HOME = os.path.expanduser('~')

def get_settings(package):
	fn = os.path.join(HOME,'.awrams',package+'.py')
	if not os.path.exists(fn):
		fn = os.path.join(HOME,'.awrams','settings.py')

	if os.path.exists(fn):
		return compile(open(fn).read(), fn, 'exec')
	else:
		return compile('','<string>','exec')
