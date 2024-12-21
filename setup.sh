#!/bin/sh

# set up whisper.cpp
git clone https://github.com/ggerganov/whisper.cpp.git
cd whsiper.cpp
sh ./models/download-ggml-model.sh large-v3-turbo-q5_0
make -j
cd ..

# for ollama, go to the website and install
# linux only
# curl -fsSL https://ollama.com/install.sh | sh