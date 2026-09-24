"""carrier-audit.

Importing the package loads any `.env` sitting at the project root, so a key put
in that file works from the CLI, the API server, the eval harness and the tests
alike. A real exported environment variable always takes precedence over the file.
"""

from . import env as _env

_env.load()
