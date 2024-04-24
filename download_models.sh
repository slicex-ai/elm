#!/bin/bash
git lfs install
TEMP_REPO="temp"
rm -rf ${TEMP_REPO}
MODELS_FOLDER="models"
rm -rf ${MODELS_FOLDER}
mkdir -p $MODELS_FOLDER

#repo download
git clone https://huggingface.co/slicexai/elm-v0.1_news_classification ${TEMP_REPO}
mv ${TEMP_REPO}/elm-* $MODELS_FOLDER/
rm -rf ${TEMP_REPO}

#repo download
git clone https://huggingface.co/slicexai/elm-v0.1_news_summarization ${TEMP_REPO}
mv ${TEMP_REPO}/elm-* $MODELS_FOLDER/
rm -rf ${TEMP_REPO}

#repo download
git clone https://huggingface.co/slicexai/elm-v0.1_toxicity_detection ${TEMP_REPO}
mv ${TEMP_REPO}/elm-* $MODELS_FOLDER/
rm -rf ${TEMP_REPO}

#repo download
git clone https://huggingface.co/slicexai/elm-v0.1_news_content_generation ${TEMP_REPO}
mv ${TEMP_REPO}/elm-* $MODELS_FOLDER/
rm -rf ${TEMP_REPO}

echo "Downloaded models!"
