# SliceX AI™ ELM (Efficient Language Models)
**ELM** (which stands for **E**fficient **L**anguage **M**odels) is the first version in the series of cutting-edge language models from [SliceX AI](https://slicex.ai) that is designed to achieve the best in class performance in terms of _quality_, _throughput_ & _memory_.

<div align="center">
  <img src="elm-rambutan.png" width="256"/>
</div>

ELM is designed to be a modular and customizable family of neural networks that are highly efficient and performant. Today we are sharing the first version in this series: **ELM-v0.1** models (named _Rambutan_). 

_Model:_ ELM introduces a new type of _(de)-composable LLM model architecture_ along with the algorithmic optimizations required to learn (training) and run (inference) these models. At a high level, we train a single ELM model in a self-supervised manner (during pre-training phase) but once trained the ELM model can be sliced in many ways to fit different user/task needs. The optimizations can be applied to the model either during the pre-training and/or fine-tuning stage. 

_Fast Inference with Customization:_ Once trained, the ELM model architecture permits flexible inference strategies at runtime depending on the deployment needs. For instance, the ELM model can  be _decomposed_ into smaller slices, i.e., smaller (or larger) models can be extracted from the original model to create multiple inference endpoints. Alternatively, the original (single) ELM model can be loaded _as is_ for inference and different slices within the model can be queried directly to power faster inference. This provides an additional level of flexibility for users to make compute/memory tradeoffs depending on their application and runtime needs.

- **Blog:** [Medium](https://medium.com/sujith-ravi/introducing-elm-efficient-customizable-privacy-preserving-llms-cea56e4f727d)

- **Github:** https://github.com/slicex-ai/elm

- **Demo** (try it out): https://huggingface.co/spaces/slicexai/elm-demo-v1

- **HuggingFace** (access ELM Model cards, code & app from HF): https://huggingface.co/slicexai

## ELM-v0.1 Model Release
This repository contains code to run our ELM models. The current ELM model `elm-v0.1` (named _Rambutan_) was pre-trained (an intermediate checkpoint was used) and then instruction fine-tuned for downstream tasks.

Models are located in the `models` folder. ELM models in this repository comes in three sizes (elm-1.0, elm-0.75 and elm-0.25). **All these different slices are extracted from the same ELM finetuned checkpoint for inference** and supports the following use-cases.
- news_classification
- toxicity_detection
- news_content_generation
- news_summarization

**NOTE: ELM-v0.1 is an early version finetuned from an intermediate pretrained checkpoint & without any KV caching, decoding optimizations, or quantization applied.**

## Setup ELM

### Download ELM repo with model files
```bash
git clone https://github.com/slicex-ai/elm
cd elm
sudo apt-get install git-lfs 
git lfs install
sh download_models.sh
```
For Macbook, replace `sudo apt-get install git-lfs` with `brew install git-lfs`
```note
NOTE: Please allow a few minutes for the full download of all model checkpoints.
```
### Installation
```bash
pip install -r requirements.txt
```
(Optional) Installing git-lfs without sudo,
```bash
wget https://github.com/git-lfs/git-lfs/releases/download/v3.2.0/git-lfs-linux-amd64-v3.2.0.tar.gz
tar -xzf git-lfs-linux-amd64-v3.2.0.tar.gz
PATH=$PATH:/<absolute-path>/git-lfs-3.2.0/
git lfs install
```
 
## Chat with ELM via Streamlit app <img src="chat.png" width="32"/>
```bash
streamlit run app.py --server.port 8011 
```

## How to use: Run ELM on a sample task (e.g., news classification)
Choose an ELM model slice for a specific task from the `models` directory and run inference.
```bash
python run.py <elm-model-directory>
e.g., python run.py models/elm-0.75_news_classification
``` 
Prompts for the specific tasks can be found in the corresponding checkpoint directory. See an example below from `models/elm-0.75_news_classification/example_prompts.json`.
```json
{
    "inputs": ["GM May Close Plant in Europe  DETROIT (Reuters) - General Motors Corp. &lt;A HREF=\"http://www.investor.reuters.com/FullQuote.aspx?ticker=GM.N target=/stocks/quickinfo/fullquote\"&gt;GM.N&lt;/A&gt; will likely  cut some jobs in Europe and may close a plant there as part of  a restructuring plan under development to try to return the  region to profitability, the U.S. automaker said on Wednesday."],
    "template": "[INST]Below is a news article. Please classify it under one of the following classes (World, Business, Sports, Sci/Tech). Please format your response as a JSON payload.\n\n### Article: {input}\n\n### JSON Response:[/INST]"
}
```

Running the above command returns the following response

```json
{
    "prompt": "[INST]Below is a news article. Please classify it under one of the following classes (World, Business, Sports, Sci/Tech). Please format your response as a JSON payload.\n\n### Article: GM May Close Plant in Europe  DETROIT (Reuters) - General Motors Corp. &lt;A HREF=\"http://www.investor.reuters.com/FullQuote.aspx?ticker=GM.N target=/stocks/quickinfo/fullquote\"&gt;GM.N&lt;/A&gt; will likely  cut some jobs in Europe and may close a plant there as part of  a restructuring plan under development to try to return the  region to profitability, the U.S. automaker said on Wednesday.\n\n### JSON Response:[/INST]",
    "response": "{'text_label': 'Business'}"
}
```
## (Optional) Setup docker container to run ELM
```bash
docker run --gpus all -it --shm-size=8g --name elm_inference --ulimit memlock=-1 --rm -v <elm-v0.1_code_path>:/elm-v0.1  nvcr.io/nvidia/pytorch:23.09-py3
```
