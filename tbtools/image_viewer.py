from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
from collections import defaultdict

import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging

logging.set_verbosity(logging.INFO)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--event_file', default='', type=str)
parser.add_argument('--port', default=7006, type=int)
parser.add_argument('--max_step', default=None, type=int)
parser.add_argument('--debug', action='store_true')


def iter_summary_from_event_file(event_file, max_step=None):
  """
  Iterates all the image summaries in the event_file.

  Args:
    event_file: the path to the event file.
    max_step: The maximum step to be truncated at. If None (defaults),
      iterates all the steps and summaries.
  """
  if not os.path.exists(event_file):
    raise IOError(event_file)

  for event in tf.train.summary_iterator(event_file):
    step = int(event.step)
    if step % 100 == 0:
      logging.info("Reading step {}, event_file={}".format(step, event_file))
    if max_step is not None and step > max_step:
      break

    if event.HasField('summary'):
      for value in event.summary.value:
        if not value.HasField('image'):
          continue

        #summary_tag = value.tag         # e.g. 'image/image/0'
        #image_encoded = value.image
        yield step, value


# create webserver
import flask
app = flask.Flask(__name__)

# TODO do not store them all!
summary_db = dict()
event_file = None

@app.route('/')
def index():
  response_html = ['<h1>%s</h1>' % 'TensorBoard Image Viewer']
  response_html.append('Event file: <pre style="display: inline-block">%s</pre>' % event_file)
  response_html.append('<ul>')
  for step in sorted(summary_db.keys()):
    response_html.append('<li><a href="/{step}">Step {step}</a></li>'.format(step=step))
  response_html.append('</ul>')
  summary_db.keys()

  response = flask.make_response('\n'.join(response_html))
  return response

@app.route('/<int:step>/')
def browse(step):
  # TODO: Use HTML templates
  try:
    tag_names = summary_db[step]

    response_html = ['<h1>Step %d</h1>' % step]
    response_html.append('<ul>')
    for tag in sorted(tag_names):
      response_html.append('<li><a href="{tag}">{tag}</a></li>'.format(tag=tag))
    response_html.append('</ul>')

    response = flask.make_response('\n'.join(response_html))
    return response
  except KeyError:
    flask.abort(404)

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


def main(args):
  FLAGS = args
  #logdir = os.path.expanduser(FLAGS.logdir)
  global event_file
  event_file = os.path.expanduser(FLAGS.event_file)

  # build summary_db
  for step, value in iter_summary_from_event_file(event_file, max_step=args.max_step):
    summary_db.setdefault(step, {})[value.tag] = value.image  # tf.summary.Summary.Image

  # run the webserver
  logging.info("Serving in port {} ...".format(FLAGS.port))
  app.run(host='0.0.0.0', port=FLAGS.port, debug=args.debug)

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)

# vim: set ts=2 sts=2 sw=2:
