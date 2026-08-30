"""Credential-provider implementations with lazy optional dependencies."""

from .chain import ChainedCredentialProvider
from .databricks import (
    CredentialProviderUnavailable,
    DatabricksSecretCredentialProvider,
    DatabricksSecretReference,
    DatabricksSecretsAccessor,
)
from .environment import EnvironmentCredentialProvider
from .vault import VaultCredentialProvider, VaultSecretReference

__all__ = [
    "ChainedCredentialProvider",
    "CredentialProviderUnavailable",
    "DatabricksSecretCredentialProvider",
    "DatabricksSecretReference",
    "DatabricksSecretsAccessor",
    "EnvironmentCredentialProvider",
    "VaultCredentialProvider",
    "VaultSecretReference",
]
