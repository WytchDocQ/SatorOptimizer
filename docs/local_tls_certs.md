# Local HTTPS (TLS) setup

This guide shows how to generate locally trusted TLS certificates and run the SATOR engine over HTTPS.

## 1) Install mkcert

Choose one method (Windows):

- Winget:
  ```powershell
  winget install --id FiloSottile.mkcert -e
  mkcert -install
  ```
- Chocolatey:
  ```powershell
  choco install mkcert -y
  mkcert -install
  ```
- Scoop:
  ```powershell
  scoop bucket add extras
  scoop install mkcert
  mkcert -install
  ```

> Important: After installing mkcert with winget/choco/scoop, close this PowerShell window and open a new one before running mkcert commands. Windows applies PATH changes to new shells only.

If your shell can’t find `mkcert`, open a new PowerShell window or add the Winget Links folder to your PATH for the current session:
```powershell
$links = "$env:LOCALAPPDATA\Microsoft\WinGet\Links"
if (Test-Path $links) { $env:Path += ";$links" }
```

## 2) Generate localhost certificates

From the repository root:
```powershell
mkdir certs
mkcert -cert-file certs\localhost.pem -key-file certs\localhost-key.pem localhost 127.0.0.1 ::1
```

This creates:
- `certs\localhost.pem` (certificate)
- `certs\localhost-key.pem` (private key)

Note: `.gitignore` includes `certs/` so your local keys are not committed.

## 3) Run the server with HTTPS

If you use a `.env` file (default), no manual exports are needed. Ensure it contains:
```
SATOR_ENABLE_TLS=true
SATOR_HTTP_PORT=8443
SATOR_TLS_CERT_FILE=certs/localhost.pem
SATOR_TLS_KEY_FILE=certs/localhost-key.pem
```
Start the server:

```powershell
sator-server
```

Access the API at `https://localhost:8443`.

## 4) Office.js add-in (alternative dev certs)

Office add-ins can use Microsoft’s dev certs:
```powershell
npx office-addin-dev-certs install
```
Point the engine to the generated files via `SATOR_TLS_CERT_FILE` and `SATOR_TLS_KEY_FILE` if you prefer those.

## Troubleshooting

- mkcert not found after install:
  - Open a new PowerShell window, or run:
    ```powershell
    $user = [Environment]::GetEnvironmentVariable('Path','User')
    $machine = [Environment]::GetEnvironmentVariable('Path','Machine')
    $env:Path = "$user;$machine"
    $links = "$env:LOCALAPPDATA\Microsoft\WinGet\Links"
    if (Test-Path $links) { $env:Path += ";$links" }
    ```
- FileNotFoundError on startup:
  - Check `SATOR_TLS_CERT_FILE` and `SATOR_TLS_KEY_FILE` point to existing files.
  - Use absolute paths if needed.

## Production note

For public deployment, terminate TLS at a reverse proxy (Caddy/NGINX) and forward to the engine on HTTP (`127.0.0.1:8080`). See `docs/operations.md` for examples.


