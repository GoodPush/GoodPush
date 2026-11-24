#!/bin/sh
# goodpush on Linux: run this from the unpacked folder (sh install.sh). It moves goodpush to ~/.local/share/goodpush-app,
# adds it to your applications menu and, with your OK, lets it use an ANT+ USB stick. Your data stays in
# ~/.local/share/goodpush; to remove the app: rm -r ~/.local/share/goodpush-app ~/.local/share/applications/goodpush.desktop
set -e
here=$(cd "$(dirname "$0")" && pwd)
app="$HOME/.local/share/goodpush-app"
mkdir -p "$app" "$HOME/.local/share/applications"
cp -R "$here"/. "$app"/
sed -e "s|^Exec=.*|Exec=$app/goodpush|" -e "s|^Icon=.*|Icon=$app/icon.png|" "$here/goodpush.desktop" \
    > "$HOME/.local/share/applications/goodpush.desktop"
echo "goodpush is in your applications menu (or run $app/goodpush)."
rule=/etc/udev/rules.d/ant-usb.rules
if [ ! -f "$rule" ]; then
    printf "Let goodpush use an ANT+ USB stick (asks for your password)? [y/N] "
    read -r yes
    case "$yes" in
        [yY]*) echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="0fcf", MODE="0666"' | sudo tee "$rule" >/dev/null
               sudo udevadm control --reload-rules && echo "Done: unplug the stick and plug it back in." ;;
    esac
fi
