try:
    import configparser
except ImportError:
    import ConfigParser as configparser
import os

import nengo.utils.appdirs
import nengo.version


_APPDIRS = nengo.utils.appdirs.AppDirs(
    nengo.version.name, nengo.version.author)

DEFAULTS = {
    'decoder_cache': {
        'enabled': True,
        'readonly': False,
        'size': 512 * 1024 * 1024,  # in bytes
        'path': os.path.join(_APPDIRS.user_cache_dir, 'decoders')
    }
}

DEFAULT_RC_FILES = [
    os.path.abspath(os.path.join(
        os.path.dirname(__file__), os.pardir, 'nengo-data', 'nengorc')),
    os.path.join(_APPDIRS.user_data_dir, 'nengorc'),
    os.environ['NENGORC'] if 'NENGORC' in os.environ else '',
    os.path.join(os.curdir, 'nengorc')
]


class _Runcom(configparser.SafeConfigParser):
    def __init__(self):
        # configparser uses old-style classes without 'super' support
        configparser.SafeConfigParser.__init__(self)
        self.reload_rc()

    def _clear(self):
        self.remove_section(configparser.DEFAULTSECT)
        for s in self.sections():
            self.remove_section(s)

    def _init_defaults(self):
        for section, settings in DEFAULTS.items():
            self.add_section(section)
            for k, v in settings.items():
                    self.set(section, k, repr(v))

    def reload_rc(self, filenames=None):
        if filenames is None:
            filenames = DEFAULT_RC_FILES

        self._clear()
        self._init_defaults()
        self.read(filenames)


runcom = _Runcom()
