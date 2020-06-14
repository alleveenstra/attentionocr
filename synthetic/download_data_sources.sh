#!/usr/bin/env bash

if [ ! -d fonts ]; then
  git clone https://github.com/google/fonts.git fonts
  find fonts -type f -iname "*.ttf" -exec mv {} fonts \;
fi

if [ ! -d texts ]; then
  mkdir -p texts
  python3 download_texts.py
fi

if [ ! -d images ]; then
  mkdir -p images
  wget https://thewallpaper.co//wp-content/uploads/2016/10preview/free-desktop-backgrounds-1920x1200-background-photos-windows-apple-mac-wallpapers-tablet-artworks-high-definition-free-1920x1200.jpg -O images/1.jpg
  wget https://www.hdwallback.net/wp-content/uploads/2018/01/Awesome-Night-Time-Wide-Hd-New-Best-Desktop-Background-Full-Free-Hd-Wallpaper-Artworks-Desktop-Images-For-Apple.jpg -O images/2.jpg
fi
