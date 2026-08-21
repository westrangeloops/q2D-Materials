# GitHub Actions Workflows for q2D-Materials

This directory contains automated workflows for building, testing, and caching the q2D-Materials development environment.

## Workflows

### 1. `cachix-push.yml` - Build & Cache
Automatically builds and pushes pre-compiled binaries to Cachix.

**When it runs:**
- Every push to `main` or `develop` branches
- Every pull request
- Weekly (Sundays at 00:00 UTC)

**What it does:**
- Builds the devenv shell on Ubuntu and macOS
- Automatically pushes successful builds to Cachix cache
- Runs the full test suite
- Verifies module imports

**Benefits for contributors:**
- Pre-compiled binaries available immediately after CI passes
- New contributors get 5-minute setup instead of 30+ minutes
- Consistent environment across platforms

### 2. `tests.yml` - Pull Request Tests
Runs comprehensive tests on pull requests.

**When it runs:**
- On all pull requests to `main` or `develop`
- On pushes to `main` or `develop`

**What it does:**
- Builds devenv shell with pre-compiled binaries from cache
- Runs full pytest suite with coverage
- Runs type checking (mypy)
- Verifies all imports work correctly
- Runs on both Ubuntu and macOS

**Benefits:**
- Ensures contributions don't break existing functionality
- Catches issues early across different platforms
- Uses cached binaries for speed

---

## How devenv + Cachix Integration Works

### Step-by-step Process

1. **Workflow triggers** (e.g., on push to main)
2. **Actions checks out code** and installs Nix
3. **Cachix action** configures the binary cache
4. **devenv is installed** into the Nix profile
5. **All subsequent steps** use `devenv shell` as the default shell
6. **Commands run** with pre-compiled dependencies from cache
7. **Cachix automatically intercepts** build outputs
8. **Successful builds pushed** back to q2d-materials cache
9. **Next contributor** automatically gets these binaries

### Key Configuration

```yaml
defaults:
  run:
    shell: devenv shell bash -- -e {0}  # All steps use devenv shell
```

This ensures:
- ✅ Consistent environment across all steps
- ✅ Pre-compiled binaries are used
- ✅ Automatic caching of new builds

### Important Notes

- **Install step must use bash**: The devenv install step must explicitly use `shell: bash` to avoid circular dependency
- **Matrix builds**: Workflows test on both `ubuntu-latest` and `macos-latest` for compatibility
- **Timeouts**: Set to 120 minutes to allow for first builds (subsequent builds are much faster)
- **Auth token required**: `CACHIX_AUTH_TOKEN` secret must be set for pushing builds

---

## Setting Up GitHub Secrets

To enable automatic pushing to Cachix:

1. Go to Repository Settings → Secrets and Variables → Actions
2. Click **New repository secret**
3. Create with:
   - **Name**: `CACHIX_AUTH_TOKEN`
   - **Value**: Your Cachix auth token from https://cachix.org/cache/q2d-materials/settings

Without this token, workflows will still work (they'll use cache for pulls only, no pushes).

---

## Workflow Status

Check workflow status:
- **Actions tab**: https://github.com/simcomat/q2D-Materials/actions
- **Cachix cache**: https://cachix.org/cache/q2d-materials
- **Branch protection**: Main branch requires all checks to pass

---

## Debugging Workflows

### View workflow logs
1. Go to Actions tab
2. Click on the failed workflow
3. Click on the failed job to see detailed logs

### Common issues

**Issue**: "Command not found: devenv"
- **Solution**: The install step didn't run. Check that it uses `shell: bash`

**Issue**: "Cannot push to cache"
- **Solution**: Check `CACHIX_AUTH_TOKEN` secret is set and valid
- **Re-generate token**: Visit https://cachix.org/cache/q2d-materials/settings

**Issue**: "Timeout waiting for build"
- **Solution**: First build takes longer. Check logs. Subsequent builds use cache and are much faster.

**Issue**: "Out of cache storage"
- **Solution**: Visit https://cachix.org/cache/q2d-materials/settings to manage cache size

---

## References

- **devenv GitHub Actions**: https://devenv.sh/integrations/github-actions/
- **Cachix Actions**: https://github.com/cachix/cachix-action
- **GitHub Actions Docs**: https://docs.github.com/en/actions

---

## Examples of devenv + Cachix in Production

These projects use similar setups:
- https://github.com/cachix/devenv
- https://github.com/numtide/flake-utils
- Many other open-source projects

---

## Q&A

**Q: Why run tests in GitHub Actions?**
A: Catch bugs early on multiple platforms before merge

**Q: Why cache builds?**
A: Contributors get 5-minute setup instead of 30+ minutes

**Q: Do I need to push manually?**
A: No! Cachix action handles it automatically

**Q: Can I test the workflow locally?**
A: Yes! Use `act` tool: `act -j test` to test locally

---

## Contributing to Workflows

If you modify `devenv.nix`:
1. Test locally: `devenv shell && devenv test`
2. Commit changes
3. Workflows automatically test on GitHub
4. New builds are cached for everyone

---

For detailed documentation, see:
- `dev/CACHIX_SETUP.md` - Cachix & devenv maintenance
- `dev/CONTRIBUTING.md` - Contribution guidelines
- `dev/ARCHITECTURE.md` - Environment architecture

