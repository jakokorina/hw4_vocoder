# TTS project(part 2, vocoder)

## Installation guide

1. Clone this repository

```shell
git clone https://github.com/jakokorina/hw4_vocoder.git
cd hw4_vocoder 
```

2. Install requirements

```shell
pip install -r ./requirements.txt
```

3. Download my model checkpoint

```python
import gdown

url = "https://drive.google.com/u/0/uc?id=1zdxuCP1-szvx5TkKmSsAgAAQYE6VjuPx"
output = "model.pth"
gdown.download(url, output, quiet=False)
```

## Results

They are located in the `results` folder. The report is [here](it_will_be_here).

## Reproducing

- Training

```shell
python3 train.py -c hw_tts/config.json
```

- Testing

```shell
python3 test.py -c hw_tts/config.json -r model.pth
```

