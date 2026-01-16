# q2D-Materials + Cachix Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                      q2D-Materials Ecosystem                         │
└─────────────────────────────────────────────────────────────────────┘

                          ┌──────────────────┐
                          │   GitHub Repo    │
                          │  (main/develop)  │
                          └────────┬─────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
                    ▼              ▼              ▼
            ┌─────────────┐  ┌──────────┐  ┌──────────────┐
            │  devenv.nix │  │.cachix.json  │
            │   (Env)     │  │(Config)  │  │ (Cache Ref)  │
            └─────────────┘  └──────────┘  └──────────────┘
                    │
                    ▼
        ┌─────────────────────────────┐
        │  GitHub Actions Workflows   │
        │                             │
        ├─ cachix-push.yml (Build)   │
        ├─ tests.yml (Test)          │
        └────────────┬────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
    ┌─────────┐          ┌──────────────────┐
    │  Build  │          │  Push to Cache   │
    │ & Test  │          │  (Cachix.org)    │
    └────┬────┘          └────────┬─────────┘
         │                        │
         └────────────┬───────────┘
                      ▼
            ┌─────────────────────────┐
            │   Cachix Cache Server   │
            │  (q2d-materials)        │
            │                         │
            │ ┌─────────────────────┐ │
            │ │ Pre-compiled        │ │
            │ │ Binaries            │ │
            │ │ (numpy, scipy, etc) │ │
            │ └─────────────────────┘ │
            └────────────┬────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
    ┌─────────┐    ┌─────────┐    ┌─────────┐
    │Developer│    │Developer│    │Developer│
    │  Mach A │    │  Mach B │    │  Mach C │
    │ (Mac)   │    │(Linux)  │    │ (WSL2)  │
    │         │    │         │    │         │
    │devenv   │    │devenv   │    │devenv   │
    │shell    │    │shell    │    │shell    │
    │(5 min)  │    │(5 min)  │    │(5 min)  │
    └─────────┘    └─────────┘    └─────────┘
```

---

## Data Flow Diagram

### Build & Cache Pipeline

```
Developer Edits Code
        │
        ▼
   Push to GitHub (main/develop)
        │
        ▼
GitHub Actions Triggered
        │
        ├─→ checkout@v4
        ├─→ install-nix-action
        ├─→ cachix-action (configure cache)
        ├─→ nix profile install devenv
        │
        ▼
Build devenv shell
        │
        ├─→ Check Cachix for: numpy, scipy, pymatgen, rdkit, etc.
        │
        ├─→ For each dependency:
        │    ├─ If in cache: Download (instant)
        │    └─ If not: Build from source, cache result
        │
        ▼
Run Tests & Verification
        │
        ├─→ pytest tests/
        ├─→ Verify imports
        ├─→ Type checking (optional)
        │
        ▼
Success ✓
        │
        ▼
Cachix Action Auto-Pushes
        │
        ├─→ All built artifacts pushed
        ├─→ Binaries cached globally
        ├─→ Public key verified
        │
        ▼
┌────────────────────────────────┐
│   Cachix Cache Updated         │
│                                │
│ q2d-materials cache now has:   │
│  ✓ numpy from ubuntu-latest    │
│  ✓ scipy from ubuntu-latest    │
│  ✓ pymatgen (NEW)              │
│  ✓ rdkit (NEW)                 │
│  ✓ And 50+ other packages      │
└────────────────────────────────┘
        │
        ▼
All future contributors:
        │
        ├─→ devenv shell
        ├─→ Cachix cache checked first
        ├─→ Pre-built binaries downloaded
        └─→ Setup in 5 minutes! ✨
```

---

## Contributor Workflow

```
┌─────────────────────────────────────────┐
│   New Contributor Arrives               │
└─────────────────────────────────────────┘
             │
             ├─ "I want to develop on q2D-Materials"
             │
             ▼
┌──────────────────────────────────────────────┐
│ Step 1: Install Nix (once per machine)       │
│ $ bash <(curl -L nixos.org/nix/install)      │
│ Time: 10 min (one-time)                      │
└──────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────┐
│ Step 2: Install devenv (once per machine)    │
│ $ nix-env -iA devenv -f ...                  │
│ Time: 2 min (one-time)                       │
└──────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────┐
│ Step 3: Clone Repository                     │
│ $ git clone https://github.com/.../q2D...    │
│ Time: 2 min                                  │
└──────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────┐
│ Step 4: Enter Development Environment        │
│ $ cd q2D-Materials                           │
│ $ devenv shell                               │
│                                              │
│ This triggers:                               │
│  1. Read devenv.nix                          │
│  2. Check .cachix.json for cache location    │
│  3. Query Cachix cache for binaries          │
│  4. Download pre-compiled packages           │
│  5. Set up environment variables             │
│  6. Create virtual environment               │
│                                              │
│ Time: 5 min (using pre-compiled binaries)    │
│ OR:  30+ min (building from source)          │
└──────────────────────────────────────────────┘
             │
             ▼
   ✨ Development Ready ✨
             │
             ├─ Write code
             ├─ Run tests: pytest tests/
             ├─ Try examples: python Examples/plot.py
             └─ Commit and push!
```

---

## File Dependencies

```
┌──────────────────────────────────────────────────────────────┐
│                    Project Root                              │
└──────────────────────────────────────────────────────────────┘
            │
    ┌───────┼────────────────────┬──────────────────┐
    │       │                    │                  │
    ▼       ▼                    ▼                  ▼
┌────────────────┐    ┌──────────────────┐    ┌──────────┐
│  devenv.nix    │    │   .cachix.json   │
│                │    │                  │
│ Declares:      │    │ References:      │
│ • Python 3.12  │    │ • Cache name     │
│ • 50+ packages │◄───┤ • Public key     │
│ • Scripts      │    │ • Fallback cache │
│ • Cachix push  │    │                  │
└────────┬───────┘    └────────┬─────────┘    └────┬─────┘
         │                     │                   │
         │                     │                   │
    ┌────┴─────────────────────┴───────────────────┴────┐
    │                                                     │
    ▼                                                     ▼
┌──────────────────────────┐                ┌──────────────────────────┐
│  Local: devenv shell     │                │  GitHub Actions          │
│                          │                │                          │
│ 1. Reads devenv.nix      │                │ Triggers via:            │
│ 2. Reads .cachix.json    │                │ • Push to main/develop   │
│ 3. Checks Cachix cache   │                │ • PR to main/develop     │
│ 4. Downloads binaries    │                │ • Weekly schedule        │
│ 5. Sets up environment   │                │                          │
│                          │                │ Uses:                    │
│ Result: Ready to develop │                │ • cachix-push.yml        │
│                          │                │ • tests.yml              │
└──────────────────────────┘                └────────────┬─────────────┘
         │                                               │
         │                                               │
         └───────────────────┬──────────────────────────┘
                             │
                             ▼
                  ┌──────────────────────────┐
                  │  Cachix Cache Network    │
                  │  (cachix.org)            │
                  │                          │
                  │ Stores & serves:         │
                  │ • Pre-compiled binaries  │
                  │ • Build artifacts       │
                  │ • For: q2d-materials    │
                  │                          │
                  │ Accessible to:           │
                  │ • Contributors (pull)    │
                  │ • Maintainers (push)     │
                  └──────────────────────────┘
```

---

## Component Interactions

### devenv.nix Lifecycle

```
When user runs: $ devenv shell

1. devenv Parses devenv.nix
   │
   ├─ Sees: packages = [ pkgs.python312, pkgs.numpy, ... ]
   ├─ Sees: cachix.pull = [ "q2d-materials" ]
   ├─ Sees: cachix.push = "q2d-materials"
   └─ Sees: scripts { dev.exec = "...", test.exec = "..." }
   │
   ▼

2. Queries Cachix Cache
   │
   ├─ Checks: https://q2d-materials.cachix.org
   ├─ For each package, asks: "Have you built this?"
   ├─ If YES: "Give me the binary"
   │   └─ Downloads pre-compiled (seconds)
   ├─ If NO: "I'll build it locally"
   │   └─ Compiles from source (minutes)
   │
   ▼

3. Sets Up Environment
   │
   ├─ Creates isolated shell
   ├─ Sets PATH, PYTHONPATH, LD_LIBRARY_PATH
   ├─ Creates virtual environment
   ├─ Installs requirements.txt
   │
   ▼

4. Runs enterShell Script
   │
   ├─ Executes: echo "✨ q2D-Materials development environment ready!"
   │
   ▼

5. User Can Now:
   ├─ Run: $ dev (initializes environment)
   ├─ Run: $ test (runs pytest)
   ├─ Run: $ python (Python 3.12 ready)
   ├─ Run: $ jupyter (Jupyter notebook)
   └─ Continue development...
```

### GitHub Actions Lifecycle

```
When code is pushed to GitHub:

1. Workflow Triggers
   │
   ├─ Event: push to main/develop
   ├─ Event: PR to main/develop
   ├─ Schedule: Weekly (0 0 * * 0)
   │
   ▼

2. GitHub Actions Starts Job
   │
   ├─ Checks out code
   ├─ Installs Nix
   ├─ Configures Cachix auth
   ├─ Installs devenv
   │
   ▼

3. Sets Default Shell
   │
   ├─ defaults.run.shell = "devenv shell bash -- -e {0}"
   ├─ All steps now use devenv shell
   ├─ Automatically queries Cachix for binaries
   │
   ▼

4. Builds & Tests
   │
   ├─ run: devenv test
   ├─ run: pytest tests/
   ├─ run: python -c "import q2D_Materials"
   │
   ▼

5. Cachix Auto-Push
   │
   ├─ Cachix action watches build
   ├─ Intercepts all build outputs
   ├─ Signs with public key
   ├─ Pushes to cache server
   │
   ▼

6. Cache Updated
   │
   ├─ New/updated binaries available
   ├─ Statistics updated
   ├─ Next contributor gets faster setup
   │
   ▼

7. Success Summary
   │
   ├─ Workflow completes ✓
   ├─ All tests pass ✓
   ├─ Binaries cached ✓
   │
   ▼

8. Contributors Benefit
   ├─ Visit cache: https://cachix.org/cache/q2d-materials
   ├─ See new builds added
   ├─ Next $ devenv shell is faster
   └─ Everyone wins! 🎉
```

---

## Deployment Model

```
                    ┌─────────────────────┐
                    │  Cachix.org Server  │
                    │ (Binary Cache CDN)  │
                    └────────────┬────────┘
                                 │
                     ┌───────────┴───────────┐
                     │                       │
          ┌──────────▼──────────┐   ┌────────▼──────────┐
          │ Binary Downloads    │   │ Statistics        │
          │ (From Cache)        │   │ Cache Hits/Miss   │
          │                     │   │ Storage Usage     │
          │ Speed: Fast!        │   │ Build History     │
          │ Size: Pre-compiled  │   │ Performance       │
          └────────────────────┘   └───────────────────┘
                     ▲
                     │
    ┌────────────────┼────────────────┐
    │                │                │
    │         ┌──────┴──────┐         │
    │         │ Maintainer  │         │
    │         │ Push (auth) │         │
    │         └─────────────┘         │
    │                │                │
    │         ┌──────▼──────┐         │
    │         │ New builds  │         │
    │         │ from CI/CD  │         │
    │         └─────────────┘         │
    │                                  │
    ▼ Read (public)                    ▼ Write (private)
┌──────────────┐                    ┌──────────────────┐
│ Contributors │                    │  Maintainers     │
│              │                    │                  │
│ Clone repo   │                    │ Edit devenv.nix  │
│ devenv shell │◄───Cachix Cache───┤ Push to GitHub   │
│ Get binaries │                    │ CI/CD runs       │
│ Setup fast! │                    │ Push to cache    │
└──────────────┘                    └──────────────────┘
```

---

## Security Architecture

```
┌──────────────────────────────────────────────────────────┐
│              Security & Trust Model                       │
└──────────────────────────────────────────────────────────┘

1. Binary Signing
   ┌──────────────────────────────────────────┐
   │ Each binary is:                          │
   │ • Hash-verified (content-addressed)      │
   │ • Signed with cache public key           │
   │ • Verified before use                    │
   │                                          │
   │ Chain of trust:                          │
   │ GitHub Actions ──→ Build  ──→ Cachix    │
   │      ↓                                    │
   │   Source Code is same as public repo     │
   │   Binaries match publicly visible source │
   │   No backdoors inserted                  │
   └──────────────────────────────────────────┘

2. Authentication
   ┌──────────────────────────────────────────┐
   │ Push Access (Maintainers Only):          │
   │ • CACHIX_AUTH_TOKEN (private key)        │
   │ • Stored in GitHub secrets               │
   │ • Never exposed in logs                  │
   │ • Read-only for contributors             │
   │                                          │
   │ Pull Access (Everyone):                  │
   │ • Public key in .cachix.json             │
   │ • Anyone can verify binaries             │
   │ • No auth required to download           │
   └──────────────────────────────────────────┘

3. Privacy
   ┌──────────────────────────────────────────┐
   │ Public Cache (q2d-materials):            │
   │ • Cache content is public                │
   │ • Same as source code availability       │
   │ • Usage stats visible to maintainers     │
   │ • No sensitive data in binaries          │
   └──────────────────────────────────────────┘
```

---

## Performance Model

```
┌──────────────────────────────────────────────────────────┐
│           Setup Time Comparison                          │
└──────────────────────────────────────────────────────────┘

Traditional pip:
  $ python3 -m venv venv         [2 min]
  $ source venv/bin/activate
  $ pip install -r requirements  [28 min] ← Compiling from source!
  ─────────────────────────────────────
  Total: 30 minutes
  Result: Compiled on your machine

devenv + Cachix:
  $ devenv shell                 [5 min] ← Pre-compiled binaries!
  ─────────────────────────────────────
  Total: 5 minutes
  Result: Pre-compiled from cache
  
Savings: 25 minutes per developer, per setup!

For team of 10:
  Traditional: 30 min × 10 = 300 min/year per feature branch
  devenv+cachix: 5 min × 10 = 50 min/year per feature branch
  
  Savings: 250 minutes/year = ~4 hours/year per developer
  For 10 developers: 40 hours/year! 🚀
```

---

## Data Storage

```
Storage Locations:

Local Machine:
  ~/.cache/nix/      ← Downloaded binaries from cache
  .venv/             ← Python virtual environment
  Total: ~1-2 GB (shared across all projects)

Cachix Cloud:
  q2d-materials.cachix.org/  ← Server-side binary cache
  Total: ~500 MB - 2 GB (shared globally)

GitHub:
  .github/workflows/  ← Workflow definitions
  devenv.nix          ← Environment declaration
  .cachix.json        ← Cache reference
  Total: ~20 KB

Source:
  q2D_Materials/      ← Your code
  Total: ~50 MB
```

---

## References

- **Official Nix**: https://nixos.org
- **devenv**: https://devenv.sh
- **Cachix**: https://cachix.org
- **devenv + GitHub Actions**: https://devenv.sh/integrations/github-actions/

---

Last Updated: January 2026


