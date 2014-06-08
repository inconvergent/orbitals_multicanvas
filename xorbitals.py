#!/usr/bin/python
# -*- coding: utf-8 -*-

import redis
from multiprocessing import Process

import cairo, Image
import numpy as np
from json import loads,dumps

from itertools import product

from numpy import sin, cos, pi, arctan2, square,sqrt, logical_not,\
                  linspace, array, zeros
from numpy.random import random, randint, shuffle


PI = pi
TWOPI = pi*2.

PORT = 6379
HOST = 'localhost'

FILEPATH = './img/test'
#COLOR_PATH = 'color/dark_cyan_white_black.gif'

NUM = 200 # number of nodes
MAXFS = 10 # max friendships pr node

CANVAS_SIZE = 5000 # size of png image
GRID_SIZE = 2

DRAW_ITT = 1000

BACK = [1.]*3
FRONT = [0,0,0,0.05]
GRAINS = 40
ALPHA = 0.05 # opacity of drawn points
STEPS = 10000

STP = 1./float(CANVAS_SIZE*GRID_SIZE)/5.

RAD = 0.26 # radius of starting circle
FARL  = 0.13 # ignore "enemies" beyond this radius
NEARL = 0.02 # do not attempt to approach friends close than this

FRIENDSHIP_RATIO = 0.1 # probability of friendship dens
FRIENDSHIP_INITIATE_PROB = 0.05 # probability of friendship initation attempt

class MultiCanvas(object):

  def __init__(self, rds,canvas_size,grid_size,filepath):

    self.grid_size = grid_size
    self.canvas_size = canvas_size
    self.filepath = filepath
    self.rds = rds

    self.__init_canvases()
    self.__start_canvases()

  def __init_canvases(self):

    self.canvases = []
    self.num_canvases = self.grid_size**2
    for i in xrange(self.num_canvases):

      render = XRender(self.canvas_size,self.rds,i,self.filepath)
      xrender = Process(target=render.run)

      self.canvases.append(xrender)

  def __start_canvases(self):

    for xrender in self.canvases:
      xrender.start()

  def dots(self,x,y,itt):

    xx = array(x,'float')
    yy = array(y,'float')
    ii = (array(xx)*self.grid_size).astype('int')
    jj = (array(yy)*self.grid_size).astype('int')
    cc = ii + jj*self.grid_size

    h = 1./self.grid_size

    for (i,j) in product(xrange(self.grid_size),repeat=2):
      ch = i + j*self.grid_size
      mask = ch == cc
      xx_trans = (xx[mask]/h)-i
      yy_trans = (yy[mask]/h)-j
      data = { 'action':'DRAW',\
               'x':list(xx_trans),\
               'y':list(yy_trans),\
               'itt':itt }
      self.rds.rpush('c{:02d}'.format(ch),dumps(data))

  def write_all(self,itt=0):
    for ch in xrange(self.num_canvases):
      self.rds.rpush('c{:02d}'.format(ch),dumps({'action':'WRITE','itt':itt}))

  def stop_now(self,itt=0):
    for ch in xrange(self.num_canvases):
      self.rds.lpush('c{:02d}'.format(ch),dumps({'action':'KILL','itt':itt}))

  def stop(self,itt=0):
    for ch in xrange(self.num_canvases):
      self.rds.rpush('c{:02d}'.format(ch),dumps({'action':'KILL','itt':itt}))
 
class XRender(object):

  def __init__(self,canvas_size,rds,channel,path):

    self.filename = path+'_{:08d}_c{:02d}.png'
    self.channel = channel

    self.canvas_size = canvas_size
    self.one = 1./canvas_size

    self.rds = rds

    self.rds.delete('c{:02d}'.format(self.channel))

    self.__init_cairo()

  def work(self, d):
    
    ctx = self.ctx
    for x,y in zip(d['x'],d['y']):
      ctx.rectangle(x,y,self.one,self.one)
      ctx.fill()

  def __init_cairo(self):

    sur = cairo.ImageSurface(cairo.FORMAT_ARGB32,self.canvas_size,self.canvas_size)
    ctx = cairo.Context(sur)
    ctx.scale(self.canvas_size,self.canvas_size)
    ctx.set_source_rgb(*BACK)
    ctx.rectangle(0,0,1,1)
    ctx.fill()

    ctx.set_source_rgba(*FRONT)

    self.sur = sur
    self.ctx = ctx

  def run(self):

    while True:

      k,v = self.rds.blpop('c{:02d}'.format(self.channel))

      d = loads(v)
      action = d['action']

      try:
        itt = d['itt']
        if not itt%100:
          print 'draw',itt, self.rds.llen('c{:02d}'.format(self.channel))
      except KeyError:
        pass

      if action == 'DRAW':

        self.work(d)

      elif action == 'WRITE':
        itt = d['itt']
        fn = self.filename.format(itt,self.channel)
        print 'writing channel:',self.channel,fn
        self.sur.write_to_png(fn)
        continue

      elif action == 'KILL':
        print self, 'done!'
        break


def connections(mc,itt,X,Y,F,A,R):

  indsx,indsy = F.nonzero()
  mask = indsx >= indsy 
  xx = []
  yy = []
  for i,j in zip(indsx[mask],indsy[mask]):
    a = A[i,j]
    d = R[i,j]
    scales = random(GRAINS)*d
    xp = X[i] - scales*cos(a)
    yp = Y[i] - scales*sin(a)
   
    #r,g,b = self.colors[ (i*NUM+j) % self.n_colors ]
    #self.ctx.set_source_rgba(r,g,b,ALPHA)
    xx.extend(xp)
    yy.extend(yp)
  
  mc.dots(xx,yy,itt)

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

  rds = redis.Redis(host=HOST,port=PORT)
  MC = MultiCanvas(rds,CANVAS_SIZE,GRID_SIZE,FILEPATH)

  X = zeros(NUM,'float')
  Y = zeros(NUM,'float')
  SX = zeros(NUM,'float')
  SY = zeros(NUM,'float')
  R = zeros((NUM,NUM),'float')
  A = zeros((NUM,NUM),'float')
  F = zeros((NUM,NUM),'byte')

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

        connections(MC,itt,X,Y,F,A,R)

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

