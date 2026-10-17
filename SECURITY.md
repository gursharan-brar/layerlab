# Security Policy

## Supported versions

layerlab is pre-alpha. Only the latest release and the `main` branch receive security fixes.

## Reporting a vulnerability

**Do not open a public issue for a security problem.**

Report it privately using GitHub's private vulnerability reporting: open the repository's **Security** tab and choose **Report a vulnerability**. Include:

- a description of the issue and its impact,
- steps or a proof of concept to reproduce it,
- the affected version or commit, and your OS and Python version.

We aim to acknowledge reports within 7 days and to agree on a fix and disclosure timeline with you.

## Scope notes

layerlab processes untrusted media. The areas most likely to matter are:

- **Input files** passed to ffmpeg and other media decoders.
- **Model weights** loaded by backends. Prefer `safetensors` or other formats that do not execute code on load. Do not load pickled checkpoints from untrusted sources.
- **Subprocess calls** to external tools such as ffmpeg. Arguments must never be built by concatenating untrusted strings into a shell command.

layerlab is designed to run locally and does not send media to remote services. A report that shows it doing so is in scope.
