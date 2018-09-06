Tensorboard Tools (`tbtools`)
=============================

This is a random collection of utilities for inspecting TensorFlow summary files.

*Under development (so for personal use at this point)!*

### Install

```
pip install -I git+https://github.com/wookayin/tensorboard-tools
```

## `tb`: Launch TensorBoard

The CLI arguments of `tensorboard` sucks. We can launch it more easily:

```
tb logdir/1 logdir/shell_expansion/*
tb --port 6006 logdir/*
```

Automatically scan train_dirs from currently running TensorFlow processes:

```
tb --auto
tb --auto <pattern>
```

## Image Viewer

Image summary in Tensorboard limits the number of steps being displayed. Run a server that would parse an event file:

```
python -m tbtools.image_viewer --logdir /tmp/path/to/traindir
python -m tbtools.image_viewer --event_file /tmp/path/to/traindir/events.out.tfevents.1234567890
```

Then we can inspect the images that are stored as image summary:

```
http://localhost:7006/<step>/<summary_tag>
http://localhost:7006/0/input_image/image/0
```


## TODO

- Advanced Image Viewer
- Draw matplotlib plots for scalar values (e.g. loss)
