# KeywordSpottingKD
## Prerequisites
```
git clone https://github.com/VSainteuf/metric-guided-prototypes-pytorch
cd metric-guided-prototypes-pytorch/
pip install -e .
cd ../
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu126.html
```

## Pull and process dataset
```
cd root/KeywordSpottingKD
python preprocess_speech_commands.py
```

## Train model
```
cd root/KeywordSpottingKD
python train_learnt_audio_prototypes.py
```