"""Provider adapters for the LLM layer.

Each module here implements the ``ProviderAdapter`` protocol from ``base`` for
one provider family, hiding its SDK's request shape, response parsing, and
usage extraction behind the shared messages-first interface.
"""
