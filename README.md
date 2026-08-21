<p align="center">
  <img src="https://github.com/vr-oj/braid/blob/main/resources/icon_128.png?raw=true" alt="BRAID Logo">
</p>

<h3 align="center">BRAID</h3>

<p align="center">
  Data processing app custom-made for BUTI data.
  <br>
  <br>
  <a href="https://pubmed.ncbi.nlm.nih.gov/41690611/">BUTI Device</a>
  -
  <a href="http://www.tykocki-lab.com/">Tykocki Lab</a>
</p>


## BRAID
BUTI Research Analysis & Inspection Dashboard

A cross-platform application for performing mechanical analyses of BUTI experimental data.

## Windows installation

1. Download `BRAID_Setup_<version>.exe` from the official GitHub release.
2. Run the setup wizard and choose the installation folder.
3. Launch BRAID from the Start menu or the optional desktop shortcut.

The installer adds BRAID to Windows' installed-apps list and includes an
uninstaller. It is not currently code-signed, so Windows SmartScreen may show
an unknown-publisher warning. Only run an installer downloaded from the
project's official GitHub release.

## Opening a recording from another application

BRAID accepts a generic `--open <recording-path>` launch argument for TIFF,
AVI, and MKV files. This lets acquisition tools hand BRAID an already-saved
recording path without sharing application code or creating a runtime
dependency; BRAID continues to work normally as a standalone analysis app.

When BRAID is already running, an external `--open` request is forwarded to
its main window. The initial blank analysis tab is reused when available;
otherwise the recording opens in a new analysis tab without replacing work in
progress. Launching BRAID normally without `--open` still creates another
separate window.

BRAID recognizes the optional BUTI v5.2 acquisition columns added by BURST to
new synchronized force CSVs. Preload, deformation, rates, configured cycles,
wire diameter, constant tension, and experiment type are displayed as
read-only acquisition context. Legacy five-column CSVs continue to load, and
these settings do not silently alter mechanics calculations.

## macOS installation

Download the DMG that matches the Mac: `Silicon` for Apple silicon (M-series)
or `Intel` for an Intel processor. Open the DMG and copy `BRAID.app` to the
Applications folder.

BRAID is not currently signed with an Apple Developer ID. On first launch,
macOS may block it as an app from an unidentified developer. Try to open BRAID
once, then open **System Settings > Privacy & Security**, scroll to **Security**,
and choose **Open Anyway**. Apple documents this process in
[Open a Mac app from an unknown developer](https://support.apple.com/guide/mac-help/open-a-mac-app-from-an-unknown-developer-mh40616/mac).

Only override this warning for a BRAID DMG downloaded from the project's
official GitHub release. A message that the app is damaged is not the expected
unsigned-app warning; re-download the matching architecture build and report
the problem rather than bypassing it with a Terminal command.

## Building the Windows release installer

BRAID uses `APP_VERSION` in `config.py` as its release version source. Install
the pinned dependencies and Inno Setup 6, then run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build-release.ps1
```

The release script checks the Python environment, runs the full test suite,
builds the one-folder Windows application, compiles the setup wizard, and
writes:

- `installer_output\BRAID_Setup_<version>.exe`
- `installer_output\BRAID_Setup_<version>.exe.sha256`

BRAID releases are built on the official Windows packaging computer and
uploaded manually; GitHub Actions is not used.
