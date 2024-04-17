git lfs install
TEMP_REPO="temp"
MODELS_FOLDER="models"
git clone https://huggingface.co/slicexai/elm-v0.1 ${TEMP_REPO}
mkdir -p $MODELS_FOLDER
mv ${TEMP_REPO}/models/* $MODELS_FOLDER/
rm -rf ${TEMP_REPO}
echo "Downloaded models!"