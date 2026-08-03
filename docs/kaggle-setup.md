# Kaggle credentials setup

`kagglehub` needs credentials before it can download anything.

## Where kagglehub looks

Verified against kagglehub 1.0.2, which checks these in order and uses the first hit:

1. `KAGGLE_API_TOKEN` environment variable (a token, or a path to a file holding one)
2. `~/.kaggle/access_token`
3. `~/.kaggle/access_token.txt`
4. `KAGGLE_USERNAME` **and** `KAGGLE_KEY` environment variables
5. `~/.kaggle/kaggle.json`

Set `KAGGLE_CONFIG_DIR` to move the `~/.kaggle` directory elsewhere.

Pick one method. They are not interchangeable credentials — see the warning below.

## Option A: access token (recommended)

The token file is **plain text with no extension**, containing only the token. Not JSON,
no quotes, no `key=` prefix. kagglehub calls `.strip()` on it, so a trailing newline is
harmless.

```zsh
mkdir -p ~/.kaggle && read -rs "TOKEN?Paste token: " && \
  printf '%s' "$TOKEN" > ~/.kaggle/access_token && \
  chmod 600 ~/.kaggle/access_token && unset TOKEN
```

`read` keeps the token out of your shell history. In bash:
`read -rsp "Paste token: " TOKEN`.

## Option B: legacy `kaggle.json`

1. Sign in at <https://www.kaggle.com>.
2. Go to <https://www.kaggle.com/settings/account>.
3. Under **API**, click **Create New Token**.
4. Your browser downloads `kaggle.json`, containing your username and key.

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Or export the same two values as environment variables:

```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key
```

> **The `key` in `kaggle.json` is not an access token.** Pasting it into
> `~/.kaggle/access_token` will fail authentication. Use it as `kaggle.json` or as
> `KAGGLE_USERNAME`/`KAGGLE_KEY`.

## Keep credentials out of the repository

Put them in your **home** directory. A credential file created inside the project folder
will not be found by kagglehub, since it only looks under `~/.kaggle` (or
`KAGGLE_CONFIG_DIR`). `.gitignore` covers `kaggle.json`, `.kaggle/`, and `access_token`
as a safety net, but the correct location is outside the repo entirely.

## Verify

```bash
.venv/bin/python -c "import kagglehub; print(kagglehub.whoami())"
```

Expected: `Kaggle credentials successfully validated.` followed by a dict containing your
username. If it raises `UnauthenticatedError`, the credential is missing, in the wrong
place, or of the wrong type.

## Accept the IEEE-CIS competition rules

IEEE-CIS is a **competition**, not a plain dataset. The API returns `403 Forbidden` until
you have accepted its rules once, in a browser, while signed in:

<https://www.kaggle.com/c/ieee-fraud-detection/rules>

Click **I Understand and Accept**. This is a one-time action per Kaggle account and cannot
be done through the API. Only needed before preparing `ieee_cis`.
