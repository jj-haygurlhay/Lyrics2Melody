# Lyrics to melody project

## Table of content



This repo implements a MIDI sequences generator using a RNN made with GRU units and a custom Transformer Decoder made on top of T5 Encoder by Google. The project was made as a final project for the course IFT6759 - Projets (avancés) en apprentissage automatique at Université de Montréal.

## Installation

To install the project, you can use the following command in a python3.9 environment with pip installed:

```bash
pip install -r requirements.txt
```

You also need to install the repo as a package, you can do so by running the following command:

```bash
pip install -e .
```

If you want to generate melodies in a `.wav` format, you need to install the `Fluidsynth` software and add a soundfont to it. 

## Usage

To use the project to train a model, you can run the following command:

```bash
python main.py
```

This will train the model using the configs that are located in the `./config/hyps.yaml` file. All the hyperparameters and metrics are logged in the `runs` directory and can be visualized using tensorboard.

If you want to train a RNN model, you can change the model `type` parameter in the config file to `RNN`. If you want to train a Transformer model, you can change the model `type` parameter in the config file to `Transformer`.

## Generate melodies

To generate melodies, you can run the following command:

```bash
python generate.py
```

This will generate a melody using the trained model path specified at the beginning of the script and save it in a `output` directory generated inside the model folder. If you want to generate a melody using a different model, you can change the `model_dir` variable at the beginning of the script and the `model_type` variable to "rnn" or "ransformer". The `text` variable can also be modified to generate a melody using a different text as the conditional input.

You can also change the following parameters in the script to modify the generated melodies:

- `temperature`: The temperature parameter used in the softmax function to sample the next note.
- `topk`: The number of top-k notes, durations and gaps to sample from.

By default, if you don't have the `Fluidsynth` software installed, the script will generate a MIDI file. If you have the software installed, it will also generate a `.wav` file.
