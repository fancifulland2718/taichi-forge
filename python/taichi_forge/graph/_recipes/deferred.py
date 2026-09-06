"""Cold reconstruction sources for explicitly detachable native recordings.

These are process-local provider descriptions, not executables or serialized
Python factories. Only freeze and materialization inspect them.
"""


class FrozenNativeRecipeSource:
    def materialize(self):
        """Return an executable owning its independent native resource leases."""
        raise NotImplementedError
