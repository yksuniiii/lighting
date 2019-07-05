# Copyright (c) 2015-2016 Anish Athalye. Released under GPLv3.

import argparse
import os,glob

from stylize_new3 import stylize, resize_image, imread

ROOT = os.path.dirname(__file__)
# DEFAULT_CONTENT = os.path.join(ROOT, 'data/train_imgs/')
# DEFAULT_CONTENT = os.path.join(ROOT, 'data/mid_train/')
DEFAULT_CONTENT = os.path.join(ROOT, 'data2/train_imgs/')
DEFAULT_TEST = os.path.join(ROOT, 'data2/test_imgs/')
DEFAULT_COEF = os.path.join(ROOT, 'data2/coefs/')
dshape = [256, 256, 3]

def build_parser():
    parser = argparse.ArgumentParser(description='neural sh lighting')
    parser.add_argument('-cd', '--content-dir', type=str,
                        default=DEFAULT_CONTENT,
                        help='the content image directory')
    parser.add_argument('-od', '--coef-dir', type=str, default=DEFAULT_COEF,
                        help='the coefs directory')
    parser.add_argument('-td', '--test-dir', type=str, default=DEFAULT_TEST,
                        help='the test image directory')
    parser.add_argument('-i', '--iterations', type=int, default=10,
                        help='the train iterations')
    parser.add_argument('-o', '--output', default='output/',
                        help='output directory')
    parser.add_argument('-d', '--device', default='/cpu:0',
                        help='running device')
    parser.add_argument('-c', '--ckpt-dir', default='./ckpt-dir/',
                        help='store model')
    parser.add_argument('-l', '--logs-dir', default='./logs/nn_logs/',
                        help='tensorboard logs')
    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()

    stylize(
        dshape=dshape,
        iterations=options.iterations,
        contents_dir=options.content_dir,
        coefs_dir=options.coef_dir,
        tests_dir=options.test_dir,
        ckpt_dir=options.ckpt_dir,
        output=options.output,
        logs_dir=options.logs_dir,
        device=options.device
    )




if __name__ == '__main__':
    main()
