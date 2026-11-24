; Windows installer (Inno Setup): the PyInstaller folder into your user's Programs folder (no admin needed),
; with Start menu (and optional desktop) shortcuts. Built by .github/workflows/build.yml.
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{6C0A9E3B-3F0E-4E5C-9B7E-2A1D5F8C7E41}
AppName=goodpush
AppVersion={#AppVersion}
AppPublisher=goodpush
DefaultDirName={localappdata}\Programs\goodpush
DefaultGroupName=goodpush
PrivilegesRequired=lowest
OutputBaseFilename=goodpush-setup
SetupIconFile=icon.ico
UninstallDisplayIcon={app}\goodpush.exe
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
DisableProgramGroupPage=yes
; the disclaimer as a page before installing
InfoBeforeFile=..\DISCLAIMER.md

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked
Name: "antdriver"; Description: "Set up an ANT+ USB stick now (plug it in first; you can also do it later from the Start menu)"; Flags: unchecked

[Files]
Source: "..\dist\goodpush\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion
; Zadig (https://zadig.akeo.ie, GPLv3, source: https://github.com/pbatard/libwdi) puts the WinUSB driver on the stick
Source: "zadig.exe"; DestDir: "{app}\tools"; Flags: ignoreversion
Source: "zadig-NOTICE.txt"; DestDir: "{app}\tools"; Flags: ignoreversion
; the steps, open in Notepad next to Zadig (ant_usb_setup.cmd: the Start menu entry opens both)
Source: "ant_usb_setup.txt"; DestDir: "{app}\tools"; Flags: ignoreversion
Source: "ant_usb_setup.cmd"; DestDir: "{app}\tools"; Flags: ignoreversion
Source: "licenses\GPL-3.0.txt"; DestDir: "{app}\tools"; DestName: "zadig-LICENSE.txt"; Flags: ignoreversion
; goodpush's licence and the notices of everything it bundles (notices.py)
Source: "..\LICENSE"; DestDir: "{app}"; DestName: "LICENSE.txt"; Flags: ignoreversion
Source: "..\NOTICE"; DestDir: "{app}"; DestName: "NOTICE.txt"; Flags: ignoreversion
Source: "THIRD_PARTY_NOTICES.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\DISCLAIMER.md"; DestDir: "{app}"; DestName: "DISCLAIMER.txt"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\goodpush"; Filename: "{app}\goodpush.exe"
Name: "{autodesktop}\goodpush"; Filename: "{app}\goodpush.exe"; Tasks: desktopicon
Name: "{autoprograms}\goodpush ANT+ USB setup"; Filename: "{app}\tools\ant_usb_setup.cmd"; IconFilename: "{app}\tools\zadig.exe"; Flags: runminimized; Comment: "Once, so goodpush can use your ANT+ USB stick"

[Run]
Filename: "{app}\goodpush.exe"; Description: "Open goodpush now"; Flags: nowait postinstall skipifsilent

; your data (in %APPDATA%\goodpush) stays when you uninstall

[Code]
procedure CurStepChanged(CurStep: TSetupStep);
var
  Code: Integer;
begin
  if (CurStep = ssPostInstall) and WizardIsTaskSelected('antdriver') then
  begin
    { the steps stay open in Notepad while Zadig runs }
    ShellExec('', 'notepad.exe', AddQuotes(ExpandConstant('{app}\tools\ant_usb_setup.txt')), '', SW_SHOWNORMAL, ewNoWait, Code);
    ShellExec('runas', ExpandConstant('{app}\tools\zadig.exe'), '', '', SW_SHOWNORMAL, ewWaitUntilTerminated, Code);
  end;
end;
