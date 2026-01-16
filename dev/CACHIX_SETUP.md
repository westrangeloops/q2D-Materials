# Cachix & devenv for q2D-Materials

This file captures everything maintainers and contributors need to keep the cached development workflow running.

## For contributors (install in 5 minutes)

```bash
bash <(curl -L https://nixos.org/nix/install)
nix-env -iA devenv -f https://github.com/NixOS/nixpkgs/tarball/nixpkgs-unstable
git clone https://github.com/simcomat/q2D-Materials.git
cd q2D-Materials
devenv shell
```

`devenv shell` automatically downloads pre-compiled binaries from the `q2d-materials` Cachix cache, so the Python environment is ready in about five minutes. Use the helper scripts inside the shell:

```bash
dev      # initialize helpers
test     # run the pytest suite
example  # quick functionality check
```

## For maintainers (keeping the cache healthy)

1. Create a Cachix account at https://cachix.org and add a cache named `q2d-materials`.
2. Copy the public key (`q2d-materials.cachix.org-1:…`) and paste it into `.cachix.json`:

```json
{
  "cachix": [
    {
      "name": "q2d-materials",
      "public-key": "q2d-materials.cachix.org-1:YOUR-PUBLIC-KEY"
    }
  ]
}
```

3. Grab an auth token from Cachix and store it as the GitHub secret `CACHIX_AUTH_TOKEN` (Settings → Secrets → Actions).

4. To verify locally:

```bash
cachix authtoken <your-token>
devenv shell
cachix push q2d-materials /nix/store/<output-path>  # optional, useful for debugging local builds
```

## Automation & workflows

- `cachix-push.yml` builds on every push/PR (Ubuntu + macOS) and pushes the artifacts to `q2d-materials`.
- `tests.yml` runs on PRs and pushes to `main/develop` so the cache stays fresh.
- Both workflows override the shell via `defaults.run.shell: devenv shell bash -- -e {0}` so every command runs inside the pinned environment.
- The Cachix action intercepts derivations and uploads them for future contributors.
- Use the Actions tab (https://github.com/simcomat/q2D-Materials/actions) to monitor runs and https://cachix.org/cache/q2d-materials for cache stats.

## Key configuration files

- `devenv.nix`: declares dependencies and configures `cachix.pull`/`cachix.push` to `q2d-materials`.
- `.cachix.json`: keeps the cache name and public key in sync.
- `.github/workflows/cachix-push.yml` and `.github/workflows/tests.yml`: control the CI matrix and caching behavior. See `dev/WORKFLOWS.md` for details.

## Troubleshooting highlights

- **No space left**: `nix-collect-garbage -d`
- **Cannot push to cache**: verify `CACHIX_AUTH_TOKEN`, regenerate on Cachix, rerun `cachix authtoken`.
- **Builds missing**: confirm `cachix.push = "q2d-materials"` in `devenv.nix` and that GitHub Actions runs to completion.
- **devenv missing**: reinstall via `nix-env -iA devenv -f https://github.com/NixOS/nixpkgs/tarball/nixpkgs-unstable`.

## Resources

- Cachix docs: https://cachix.org/help
- devenv docs: https://devenv.sh/
- Cachix action: https://github.com/cachix/cachix-action

Need more details? `dev/WORKFLOWS.md` walks through every job, and `dev/ARCHITECTURE.md` explains how the environment pieces fit together.
