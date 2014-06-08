#!/usr/bin/python
# -*- coding: utf-8 -*-

import Image
import redis
import os
import numpy as np
from numpy import sin,cos,pi,arctan2,square,sqrt,logical_not,\
                  linspace,array,zeros
from numpy.random import random,randint,shuffle
from multicanvas.MultiCanvas import MultiCanvas

PI = pi
TWOPI = pi*2.

PORT = 6379
HOST = 'localhost'

CANVAS_SIZE = 20000 # size of canvas
GRID_SIZE = 2 # number of canvases in each direction

NUM = 700 # number of nodes
MAXFS = 10 # max friendships pr node

DRAW_ITT = 5000

BACK = [1.]*3
GRAINS = 20
ALPHA = 0.05 # opacity of drawn points
STEPS = 10**7

STP = 1./float(CANVAS_SIZE)/15.

RAD = 0.26 # radius of starting circle
FARL  = 0.13 # ignore "enemies" beyond this radius
NEARL = 0.02 # do not attempt to approach friends close than this

FRIENDSHIP_RATIO = 0.1 # probability of friendship dens
FRIENDSHIP_INITIATE_PROB = 0.05 # probability of friendship initation attempt

COLOR_PATH = '../colors/dark_cyan_white_black.gif'

PATH = './cyan_gp_a'
PATH += '_num{:d}_fs{:d}_near{:2.4f}_far{:2.4f}_pa{:2.4f}_pb{:2.4f}_rad{:2.4f}'\
            .format(NUM,MAXFS,NEARL,FARL,FRIENDSHIP_RATIO,\
                    FRIENDSHIP_INITIATE_PROB,RAD)
PATH += '/img'

def ensure_dir(fn):
  
  dir = os.path.dirname(fn)

  try:
      os.stat(dir)
  except:
      os.mkdir(dir) 

def get_colors(f):

  scale = 1./255.
  im = Image.open(f)
  w,h = im.size
  rgbim = im.convert('RGB')
  colors = []
  for i in xrange(0,w):
    for j in xrange(0,h):
      r,g,b = rgbim.getpixel((i,j))
      colors.append((r*scale,g*scale,b*scale))

  shuffle(colors)
  return colors

def connections(mc,itt,C,X,Y,F,A,R):

  n_colors = len(C)
  indsx,indsy = F.nonzero()
  mask = indsx >= indsy 
  xx = []
  yy = []
  colors = []
  for i,j in zip(indsx[mask],indsy[mask]):
    a = A[i,j]
    d = R[i,j]
    scales = random(GRAINS)*d
    xp = X[i] - scales*cos(a)
    yp = Y[i] - scales*sin(a)
   
    r,g,b = C[(i*NUM+j) % n_colors]

    colors.extend([(r,g,b,ALPHA) for _ in xrange(GRAINS)])
    xx.extend(xp)
    yy.extend(yp)
 
  mc.dots(xx,yy,colors,itt)

def set_distances(X,Y,A,R):

  for i in xrange(NUM):

    dx = X[i] - X
    dy = Y[i] - Y
    R[i,:] = square(dx)+square(dy)
    A[i,:] = arctan2(dy,dx)

  sqrt(R,R)

def make_friends(i,F,R):

  cand_num = F.sum(axis=1)
  
  if cand_num[i]>=MAXFS:
    return

  cand_mask = cand_num<MAXFS
  cand_mask[i] = False
  cand_ind = cand_mask.nonzero()[0]

  cand_dist = R[i,cand_ind].flatten()
  cand_sorted_dist = cand_dist.argsort()
  cand_ind = cand_ind[cand_sorted_dist]

  cand_n = len(cand_ind)

  if cand_n<1:
    return

  for k in xrange(cand_n):

    if random()<FRIENDSHIP_RATIO:

      j = cand_ind[k]
      F[[i,j],[j,i]] = True
      return

def main():

  ensure_dir(PATH)

  rds = redis.Redis(host=HOST,port=PORT)

  MC = MultiCanvas(redis.Redis(host=HOST,port=PORT),\
                   CANVAS_SIZE,GRID_SIZE,PATH,BACK)

  X = zeros(NUM,'float')
  Y = zeros(NUM,'float')
  SX = zeros(NUM,'float')
  SY = zeros(NUM,'float')
  R = zeros((NUM,NUM),'float')
  A = zeros((NUM,NUM),'float')
  F = zeros((NUM,NUM),'byte')
  C = get_colors(COLOR_PATH)

  for i in xrange(NUM):
    the = random()*TWOPI
    x = RAD * sin(the)
    y = RAD * cos(the)
    X[i] = 0.5+x
    Y[i] = 0.5+y

  try:

    for itt in xrange(STEPS):

      try:

        set_distances(X,Y,A,R)
        
        SX[:] = 0.
        SY[:] = 0.
        
        for i in xrange(NUM):
          xF = logical_not(F[i,:])
          d = R[i,:]
          a = A[i,:]
          near = d > NEARL
          near[xF] = False
          far = d < FARL
          far[near] = False
          near[i] = False
          far[i] = False
          speed = FARL - d[far]

          SX[near] += cos(a[near])
          SY[near] += sin(a[near])
          SX[far] -= speed*cos(a[far])
          SY[far] -= speed*sin(a[far])

        X += SX*STP
        Y += SY*STP

        if random()<FRIENDSHIP_INITIATE_PROB:

          k = randint(NUM)
          make_friends(k,F,R)

        connections(MC,itt,C,X,Y,F,A,R)

        if not itt%100:
          print 'itteration:',itt

        if not itt%DRAW_ITT:
          if itt>0:
            MC.write_all(itt)

      except KeyboardInterrupt:
        MC.stop_now(itt)
        break

  except Exception, e:
    raise
  finally:
    MC.stop(itt)


if __name__ == "__main__":
  main()

