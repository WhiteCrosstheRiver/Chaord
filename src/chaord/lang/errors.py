"""Chaord error types."""


class ChaordError(Exception):
    """Base class for all Chaord errors."""


class ChaordSyntaxError(ChaordError):
    """Parse or structural error, carrying a 1-based source line."""
