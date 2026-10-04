# 1. Install SpatioEv

You do this **once** per computer. Afterwards, [updating](#updating-to-the-latest-version)
takes one minute.

You need about 3 GB of free disk space and an internet connection. You do
**not** need a GitHub account.

## What you are installing, and why

| Program | What it is for |
|---|---|
| **Anaconda** (or Miniconda) | Installs Python and keeps SpatioEv's software in its own *environment*, so it never clashes with other programs on your Mac. |
| **Git** | Downloads SpatioEv from GitHub, and later fetches updates. macOS installs it for you the first time it is needed. |
| **SpatioEv** | The analysis package and its app, which opens in your web browser. |
| **QuPath 0.7** | Where you classify cells ([step 5](qupath.md)). |

## Before you start

### Open Terminal

Press ++cmd+space++, type `Terminal`, and press ++return++. A window with a
text prompt ending in `%` opens. This is where you paste the commands below.

!!! tip "Keep Terminal in your Dock"
    Right-click the Terminal icon in the Dock › *Options* › *Keep in Dock*.
    You will open it every time you use SpatioEv.

### Install Anaconda (skip if you have it)

Check whether you already have it:

```bash
conda --version
```

- It prints something like `conda 24.9.2` → you have it. Go to
  [Install SpatioEv](#install-spatioev).
- It says `command not found: conda` → install it:

1. Find out which chip your Mac has:  menu › *About This Mac*. *Chip: Apple
   M1/M2/M3/M4…* means **Apple silicon**; *Processor: Intel…* means **Intel**.
2. Go to [anaconda.com/download](https://www.anaconda.com/download), download
   the **macOS graphical installer** for your chip, open it and click through
   with the default options.
3. **Quit Terminal completely** (++cmd+q++) and open it again, so it notices
   the new installation.
4. Run `conda --version` again. It should now print a version number. The
   start of your prompt shows `(base)`.

### Install QuPath 0.7 (skip if you have it)

Download QuPath **0.7** for macOS from
[qupath.github.io](https://qupath.github.io), open the `.pkg` (or drag the app
into *Applications*) and start it once. If macOS says it cannot check the app
for malicious software, open  › *System Settings* › *Privacy & Security* and
click *Open Anyway*.

## Install SpatioEv

### Step 1. Check you do not already have a SpatioEv folder

```bash
ls ~/SpatioEv
```

- `No such file or directory` → good, continue with step 2.
- It lists files → SpatioEv is already installed. Go to
  [Updating](#updating-to-the-latest-version) instead.

!!! note "What `~` means"
    `~` is short for your home folder, `/Users/<your name>`. `~/SpatioEv` is a
    folder called *SpatioEv* in it.

### Step 2. Create SpatioEv's environment

```bash
conda create -n spatioev_env python=3.11 -y
```

This makes an empty, separate Python 3.11 installation called `spatioev_env`.
It takes a minute or two and prints a lot of text; wait for the `%` prompt.

```bash
conda activate spatioev_env
```

This switches Terminal into that environment. The start of your prompt
changes from `(base)` to `(spatioev_env)`.

!!! warning "Every new Terminal window starts outside the environment"
    Whenever you open Terminal to use SpatioEv, run
    `conda activate spatioev_env` first. If you forget, you will see
    `command not found: spatioev`.

### Step 3. Download SpatioEv

```bash
cd ~ && git clone --depth 1 https://github.com/Bashford-Rogers-lab/SpatioEv.git
```

This downloads SpatioEv into `~/SpatioEv` (about 1–2 minutes).

!!! note "If a window asks to install *command line developer tools*"
    That is macOS offering to install Git. Click **Install**, wait until it
    finishes (5–15 minutes), then run the command above again.

!!! note "Why `--depth 1`"
    It downloads only the current version (about 60 MB) instead of the whole
    project history (about 700 MB), which can take very long or stall.

### Step 4. Install SpatioEv into the environment

```bash
cd ~/SpatioEv && pip install -e ".[apps]"
```

This installs SpatioEv and everything the app needs (about a minute).
`-e` means SpatioEv runs straight from the `~/SpatioEv` folder, so updating
that folder updates SpatioEv.

### Step 5. Check it worked

```bash
spatioev --help
```

You should see a short list of commands including `ui`, `qupath`, `batch`
and `demo`. If you do, SpatioEv is installed.

Continue with [Practise on demo data](demo.md) or, if you prefer to start
with your own data, [Organise your data](organise.md).

## Every time you use SpatioEv

Open Terminal and run:

```bash
conda activate spatioev_env
```

Then start the app as described in [Open the app](app.md).

## Updating to the latest version

When you are told a new version is available, run these one at a time:

```bash
conda activate spatioev_env
```

```bash
cd ~/SpatioEv && git checkout main && git pull
```

```bash
pip install -e ".[apps]"
```

The second command fetches the new code; the third installs any new software
it needs (it finishes quickly when nothing changed). Your data and results are
never touched: they live in your own folders, not in `~/SpatioEv`.

!!! note "If `git checkout` complains about local changes"
    Run `git stash`, then repeat the command. This sets aside any accidental
    edits to SpatioEv's own files.

!!! note "Installed before October 2026?"
    Older installs may be on a branch other than `main`. If `git pull` says
    *There is no tracking information*, run this once, then repeat the update:

    ```bash
    cd ~/SpatioEv && git remote set-branches origin main && git fetch origin main && git checkout main && git branch --set-upstream-to=origin/main main
    ```

## If something goes wrong

| What you see | What to do |
|---|---|
| `command not found: conda` | Anaconda is not installed, or Terminal was not restarted after installing it. Quit Terminal with ++cmd+q++ and reopen. |
| `command not found: spatioev` | Run `conda activate spatioev_env` first. |
| `destination path 'SpatioEv' already exists` | SpatioEv is already installed. Use [Updating](#updating-to-the-latest-version). |
| The download is very slow or stalls | You probably left out `--depth 1`. Press ++ctrl+c++ to stop it, delete the half-finished folder with `rm -rf ~/SpatioEv`, and repeat step 3 exactly. |
| `EnvironmentNameNotFound: spatioev_env` | Step 2 did not finish. Run it again. |

More in [Troubleshooting](troubleshooting.md).
