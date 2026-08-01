# Kaggle credentials setup

`kagglehub` needs an API token before it can download anything.

## 1. Create the token

1. Sign in at <https://www.kaggle.com>.
2. Go to <https://www.kaggle.com/settings/account>.
3. Under **API**, click **Create New Token**.
4. Your browser downloads `kaggle.json`, containing your username and key.

## 2. Install the token

Place the file where kagglehub looks for it, and restrict its permissions:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Alternatively, export the credentials as environment variables instead of using a file:

```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key
```

A third option: set `KAGGLE_CONFIG_DIR` to a directory containing `kaggle.json`.

**Never commit `kaggle.json`.** It is listed in `.gitignore`.

## 3. Verify

```bash
.venv/bin/python -c "import kagglehub; print(kagglehub.whoami())"
```

Expected: a dict containing your username. If it raises instead, the token is missing
or wrong — repeat step 1.

## 4. Accept the IEEE-CIS competition rules

IEEE-CIS is a **competition**, not a plain dataset. The API returns `403 Forbidden` until
you have accepted its rules once, in a browser, while signed in:

<https://www.kaggle.com/c/ieee-fraud-detection/rules>

Click **I Understand and Accept**. This is a one-time action per Kaggle account and cannot
be done through the API. Only needed before preparing `ieee_cis`, which arrives in the
second implementation plan.
