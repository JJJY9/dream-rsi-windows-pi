# Security and execution boundaries

This project runs model-generated Python programs and can invoke a model-powered coding agent. On native Windows, those programs and agent processes run with the current user's permissions. Workspace directories, static checks, timeouts, and Windows Job Objects constrain normal execution and resource use; they do **not** provide a filesystem or network security sandbox.

Use a separate account or machine if the model, task input, or generated code is not trusted with the files and network access available to your Windows user. Do not place API keys in task workspaces or commit them to Git. The CLI names a provider and model for each live run; credentials are configured through Pi by the user. `model-check` and `run` make real provider requests and may incur costs.

`.runtime/`, `state/`, virtual environments, local configuration, transcripts, databases, and logs are excluded from the public source. Run IDs and event records may still contain local paths or model output, so inspect any evidence before deliberately publishing it.
