# Installing SpatioEv

These instructions have moved into the **step-by-step guide**, which also
covers organising your data, QuPath and every analysis step, with
screenshots:

- On the web: [https://bashford-rogers-lab.github.io/SpatioEv/guide/install/](https://bashford-rogers-lab.github.io/SpatioEv/guide/install/)
- In this repository: [docs/guide/install.md](docs/guide/install.md)

## Updating an existing install (quick reference)

```bash
conda activate spatioev_env
```

```bash
cd ~/SpatioEv && git checkout main && git pull
```

```bash
pip install -e ".[apps]"
```

If `git pull` says *There is no tracking information*, see
[Updating](docs/guide/install.md#updating-to-the-latest-version) in the guide.
