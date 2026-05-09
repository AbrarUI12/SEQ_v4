# Third-Party Quantization Integrations

This directory isolates external quantization methods from SEQ core code.

## OmniQuant

Upstream source:

```text
https://github.com/OpenGVLab/OmniQuant
```

Pinned commit:

```text
feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14
```

Local checkout:

```text
third_party_quant/OmniQuant/
```

This clean repo is not currently a git repository, so the upstream source is stored as a nested pinned git checkout rather than a parent-repo submodule. Keep the nested checkout detached at the pinned commit unless intentionally updating the integration.

On Windows, the upstream repo has a case-only filename collision:

```text
imgs/OmniQuant.png
imgs/omniquant.png
```

Git may show one image as modified on case-insensitive filesystems. This is a checkout artifact; do not treat it as a SEQ-side algorithm edit.

## Adapter Rule

SEQ-facing code must call upstream OmniQuant through:

```text
third_party_quant/adapters/omniquant_adapter.py
```

The adapter may build commands, launch processes, collect artifacts, and write provenance. It must not reimplement OmniQuant internals.

## Environment Rule

Use a separate environment for upstream OmniQuant. See:

```text
third_party_quant/envs/omniquant-upstream.environment.yml
```

Do not merge upstream OmniQuant dependency assumptions into the main SEQ environment unless there is a deliberate compatibility pass.

