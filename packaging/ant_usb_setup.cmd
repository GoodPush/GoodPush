@echo off
rem goodpush ANT+ USB setup (Start menu): the steps in Notepad, next to Zadig (which asks Windows for permission)
start "" notepad.exe "%~dp0ant_usb_setup.txt"
start "" "%~dp0zadig.exe"
