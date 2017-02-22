from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
from collections import defaultdict

import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import event_file_inspector as efi
from tensorflow.python.summary import event_multiplexer
from tensorflow.tensorboard.backend import server

logging.set_verbosity(logging.INFO)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--event_file', default='', type=str)
parser.add_argument('--port', default=7006, type=int)


def iter_summary_from_event_file(event_file):
  if not os.path.exists(event_file):
    raise IOError(event_file)

  generator = efi.generator_from_event_file(event_file)
  for event in generator:
    step = int(event.step)
    logging.info("Reading step {}, event_file={}".format(step, event_file))
    if event.HasField('summary'):
      for value in event.summary.value:
        if not value.HasField('image'):
          continue

        #summary_tag = value.tag         # e.g. 'image/image/0'
        #image_encoded = value.image
        yield step, value


def main(args):
  FLAGS = args
  #logdir = os.path.expanduser(FLAGS.logdir)
  event_file = os.path.expanduser(FLAGS.event_file)

  # TODO do not store them all!
  summary_db = dict()

  for step, value in iter_summary_from_event_file(event_file):
    summary_db.setdefault(step, {})[value.tag] = value.image  # tf.summary.Summary.Image

  # create webserver
  import flask
  app = flask.Flask(__name__)

  @app.route('/<int:step>/<path:tag_name>')
  def get_data(step, tag_name):
    logging.debug("Requested : {}, {}".format(step, tag_name))
    try:
      image_str = summary_db[step][tag_name]
      response = flask.make_response(image_str.encoded_image_string)
      response.headers['Content-Type'] = 'image/png'
      return response
    except KeyError:
      flask.abort(404)

  logging.info("Serving in port {} ...".format(FLAGS.port))
  app.run(host='0.0.0.0', port=FLAGS.port)

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)

# vim: set ts=2 sts=2 sw=2:
