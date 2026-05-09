class Gltf2BbError(Exception):
    """Base class for user-facing gltf2bb errors."""


class InspectError(Gltf2BbError):
    """Raised for user-facing inspect failures."""


class PartitionError(Gltf2BbError):
    """Raised for user-facing partition failures."""


class ConvertError(Gltf2BbError):
    """Raised for user-facing convert failures."""


class ConfigError(Gltf2BbError):
    """Raised for user-facing preset/config failures."""
