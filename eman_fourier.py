#!/usr/bin/python
# 
# Copyright 2014 by Peter Kasson
# Except for code by Steve Ludtke as noted below

"""Perform radially averaged FFT on images."""

import gflags
import glob
import numpy
import sys

from EMAN2 import EMData


def runFFT(images, apix=3.7, do_avg=True):
  """Run radially averaged FFT on an EMAN image.
  Args:
    images: 
    apix:  angstroms per pixel
    do_avg:  compute average
  Rets:
    curve:  radially-averaged FFT
    sf_dx:  corresponding x values.
  """
  # This code by Steven Ludtke
  nx = images[0]["nx"] + 2
  ny = images[0]["ny"]
  fftavg = EMData(nx, ny, 1)
  fftavg.to_zero()
  fftavg.set_complex(1)

  for d in images:
    d.process_inplace("mask.ringmean")
    d.process_inplace("normalize")
    df = d.do_fft()
    df.mult(df.get_ysize())
    fftavg.add_incoherent(df)
  if do_avg:
    fftavg.mult(1.0 / len(images))

  curve = fftavg.calc_radial_dist(ny, 0, 0.5, 1)
  sf_dx = 1.0 / (apix * 2.0 * ny)
  return (curve, sf_dx)


def runFFT_err(images, apix=3.7, nresamples=1000, conf_int=95):
  """Run radially averaged FFT with bootstrapped error estimates.
  Args:
    images: images to operate on
    apix:  angstroms per pixel
    nresmaples: number of bootstrap resamples
    conf_int:  confidence interval to calculate
  Rets:
    curve:  average y-vals
    conf_lo:  low confidence bound
    conf_hi: high confidence bound
    sf_dx:  x-vals
  """
  bootidx = numpy.random.randint(len(images) - 1,
                                 size=(nresamples, len(images)))
  nx = images[0]["nx"]
  ny = images[0]["ny"]
  # compute FFT on everything
  # for now also computing sample average
  df = []
  sampleavg = EMData(nx + 2, ny, 1)
  sampleavg.to_zero()
  sampleavg.set_complex(1)

  for d in images:
    d.process_inplace("mask.ringmean")
    d.process_inplace("normalize")
    curimg = d.do_fft()
    curimg.mult(d.get_ysize())
    sampleavg.add_incoherent(curimg)
    df.append(curimg)
  sampleavg.mult(1.0 / len(images))

  # calculate average for each bootstrap resample
  fftavg = EMData(nx + 2, ny, 1)
  curve = []
  for r in range(nresamples):
    fftavg.to_zero()
    fftavg.set_complex(1)
    for idx in bootidx[r]:
      fftavg.add_incoherent(df[idx])
    fftavg.mult(1.0 / len(bootidx[r]))
    curve.append(fftavg.calc_radial_dist(ny, 0, 0.5, 1))

  cdata = numpy.vstack(curve)
  xdata = numpy.array(range(cdata.shape[-1])) * 1.0 / (apix * 2.0 * ny)
  return (sampleavg.calc_radial_dist(ny, 0, 0.5, 1),
          numpy.percentile(cdata, 50 - conf_int/2, axis=0),
          numpy.percentile(cdata, 50 + conf_int/2, axis=0),
          xdata)


if __name__ == '__main__':
  FLAGS = gflags.FLAGS
  gflags.DEFINE_string('infilename', '', 'input file')
  gflags.DEFINE_string('outfilename', '', 'output file or suffix if multiple')
  gflags.DEFINE_string('imgsuffix', '',
                       'image suffix to replace if multi-input')
  gflags.DEFINE_boolean('bootstrap', True, 'run bootstrap error analysis')
  argv = FLAGS(sys.argv)

  # Support multiple input files
  infile_list = glob.glob(FLAGS.infilename)
  for infile in infile_list:
    print 'Processing %s' % infile
    if len(infile_list) == 1:
      outfile = FLAGS.outfilename
    else:
      # kludge
      outfile = infile.replace(FLAGS.imgsuffix, FLAGS.outfilename)
    # read images
    input_imgs = EMData.read_images(infile)
    if FLAGS.bootstrap:
      (yvals, err_lo, err_hi, xdata) = runFFT_err(input_imgs)
      numpy.savetxt(outfile,
                    numpy.array([xdata, yvals, err_lo, err_hi]))
    else:
      (yvals, dx) = runFFT(input_imgs)
      numpy.savetxt(outfile,
                    numpy.array([numpy.arange(0, len(yvals) * dx, dx),
                                 yvals]))
