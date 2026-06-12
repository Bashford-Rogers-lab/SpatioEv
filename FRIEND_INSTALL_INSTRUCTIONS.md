# SpatioEv Fresh Install Instructions

These instructions are for macOS Terminal. Copy and paste one command block at a
time into Terminal, then press Enter.

Important: Step 2 deletes the old `SpatioEv` folder from your home folder. If
you saved any personal files inside that folder, move them somewhere else before
running Step 2.

## Step 1: Open Terminal

Open the macOS Terminal app:

1. Press `Command + Space`.
2. Type `Terminal`.
3. Press `Enter`.

You will paste all commands below into Terminal.

## Step 2: Remove the old SpatioEv folder

Paste this into Terminal:

```bash
cd ~
rm -rf SpatioEv
```

## Step 3: Remove the old conda environment

Paste this into Terminal:

```bash
conda env remove -n spatioev_env
```

If Terminal says the environment does not exist, that is okay. Continue to the
next step.

## Step 4: Download the fresh SpatioEv folder from GitHub

Paste this into Terminal:

```bash
cd ~
git clone https://github.com/Bashford-Rogers-lab/SpatioEv.git
```

## Step 5: Go into the correct SpatioEv folder

Paste this into Terminal:

```bash
cd ~/SpatioEv
pwd
ls
```

You should see this path:

```text
/Users/shiyantang/SpatioEv
```

You should also see these two files:

```text
environments.yml
requirements-spatioev_env.txt
```

If you do not see both files, stop and ask for help.

## Step 6: Create the basic conda environment

Paste this into Terminal:

```bash
conda env create -f environments.yml
```

This step may take a few minutes.

## Step 7: Activate the environment

Paste this into Terminal:

```bash
conda activate spatioev_env
```

After this, the left side of Terminal should show:

```text
(spatioev_env)
```

## Step 8: Install the pinned Python packages

Paste this into Terminal:

```bash
python -m pip install --no-deps -r requirements-spatioev_env.txt
```

This step may take a while.

## Step 9: Test that SpatioEv works

Paste this into Terminal:

```bash
python -c "import spatioev; print(spatioev.__version__)"
```

If it prints:

```text
0.1.0
```

then the install worked.

## Everyday Use

Each time you open a new Terminal window and want to use SpatioEv, paste:

```bash
cd ~/SpatioEv
conda activate spatioev_env
```

## If Something Goes Wrong

Copy the full Terminal error message and send it to Shihong.

Please also run this command and send the output:

```bash
pwd
ls
conda env list
```
