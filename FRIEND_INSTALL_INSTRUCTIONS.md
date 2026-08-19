# Installing SpatioEv

Instructions for macOS Terminal. Copy one block at a time, paste it into
Terminal, and press Enter. Wait for each to finish before the next.

You need [Anaconda or Miniconda](https://www.anaconda.com/download) installed.
You do **not** need a GitHub account or SSH keys.

---

## Part 1 — First-time install

### Step 1. Open Terminal

Press `Command + Space`, type `Terminal`, press Enter.

### Step 2. Check you don't already have a SpatioEv folder

```bash
ls ~/SpatioEv
```

- **"No such file or directory"** → good, continue to Step 3.
- **It lists files** → you already have SpatioEv. Skip to
  [Part 2 — Updating](#part-2--updating-an-existing-install) instead.

> Do not try to move an existing folder out of the way with `mv`. If a backup
> folder already exists, `mv` puts the new folder *inside* it and the download
> in Step 4 then fails with "destination path already exists".

### Step 3. Create the environment

```bash
conda create -n spatioev_env python=3.11 -y
```

```bash
conda activate spatioev_env
```

The left of your prompt should now show `(spatioev_env)`.

### Step 4. Download SpatioEv

```bash
cd ~ && git clone --depth 1 --single-branch -b reorganize-and-optimize https://github.com/Bashford-Rogers-lab/SpatioEv.git
```

This takes about 1–2 minutes.

> `--depth 1` matters. It downloads the current version only (about 56 MB)
> instead of the full project history (about 700 MB), which can take more than
> ten minutes or stall.

### Step 5. Install

```bash
cd ~/SpatioEv && pip install -e ".[apps]"
```

This takes about 30 seconds and installs everything needed for the interface.

### Step 6. Check it worked

```bash
python -c "import spatioev; print(spatioev.__version__)"
```

It should print:

```text
0.2.0
```

If it does, you are done.

---

## Part 2 — Updating an existing install

If you already have `~/SpatioEv`, update it in place. Do **not** delete or
rename the folder.

```bash
conda activate spatioev_env
```

```bash
cd ~/SpatioEv && git fetch origin && git checkout reorganize-and-optimize && git pull
```

```bash
pip install -e ".[apps]"
```

```bash
python -c "import spatioev; print(spatioev.__version__)"
```

> If `git checkout` complains about local changes, run `git stash` first, then
> repeat the command.

---

## Part 3 — Everyday use

Each time you open a new Terminal window:

```bash
conda activate spatioev_env
```

Then start the interface:

```bash
spatioev ui
```

Your browser opens at `http://localhost:8501`. To stop it, click the Terminal
window and press `Control + C`.

To point the interface at a specific project folder:

```bash
spatioev ui --project-root /path/to/your/project
```

Other options: `--port 8502` if 8501 is busy, `--no-browser` to start without
opening a browser tab.

---

## If something goes wrong

| Message | What to do |
|---|---|
| `destination path 'SpatioEv' already exists` | You already have it — use Part 2 instead of Part 1. |
| `command not found: spatioev` | Run `conda activate spatioev_env` first. |
| `command not found: conda` | Anaconda is not installed, or reopen Terminal after installing it. |
| `Port 8501 is already in use` | Something is already running. Use `spatioev ui --port 8502`. |
| Clone is very slow or stalls | You probably omitted `--depth 1`. Press `Control + C` and redo Step 4 exactly. |

If you are stuck, send Shihong the full error text plus the output of:

```bash
pwd && conda env list && python -c "import spatioev; print(spatioev.__version__)"
```

---

## Notes

- **Verified** on macOS with a clean environment: clone ~80 s, install ~23 s,
  interface reachable at `http://localhost:8501`.
- Installed alongside SpatioEv: Python 3.11, zarr 3, tifffile, pandas 3,
  scanpy, Streamlit.
- `-e` installs in *editable* mode, so after `git pull` the new code is picked
  up immediately — no reinstall needed unless dependencies changed.
- The `-b reorganize-and-optimize` flag selects the current working branch.
  Once it is merged, drop that flag and the default branch will be correct.
- Interactive Napari gating is a separate optional extra:
  `pip install -e ".[gating]"`. It is not needed for the standard workflow.
