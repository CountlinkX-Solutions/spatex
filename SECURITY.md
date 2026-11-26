# Security policy

## Supported versions

Spatex is pre-1.0. Security fixes go into the latest 0.x release only.

| Version | Supported |
|---|---|
| Latest 0.x release (currently 0.2.x) | Yes |
| Older releases | No |

## Reporting a vulnerability

Do not open a public issue. Report the vulnerability privately through GitHub's private
vulnerability reporting: open the
[Security tab of the repository](https://github.com/CountlinkX-Solutions/spatex/security) and
choose "Report a vulnerability". Include the affected version, a description of the problem and
the input or steps that reproduce it. The maintainers reply in the private advisory and
coordinate the fix and its disclosure there.

Examples of what counts as a vulnerability in Spatex:

- input that crashes the BEAM through the native code, for example an allocation that the size
  guards in `spatex_core::limits` do not catch (a failed allocation aborts the whole VM);
- a problem with the integrity of the precompiled NIF download (release artifacts or the
  checksum file shipped in the Hex package).

Wrong numerical results are bugs, not vulnerabilities: report them as an issue.
