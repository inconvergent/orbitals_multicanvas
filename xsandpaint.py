#!/usr/bin/python
# -*- coding: utf-8 -*-

import redis
import threading

import cairo, Image
import numpy as np
from json import loads,dumps

from numpy import sin, cos, pi, arctan2, square,sqrt, logical_not,\
                  linspace, array, zeros
from numpy.random import random, randint, shuffle

PI = pi
TWOPI = pi*2.

PORT = 6379
HOST = 'localhost'


#COLOR_PATH = 'color/dark_cyan_white_black.gif'

NUM = 100 # number of nodes
MAXFS = 10 # max friendships pr node

SIZE = 2000 # size of png image
DRAW_ITT = 4000

BACK = [1.]*3
FRONT = [0,0,0,0.05]
GRAINS = 40
ALPHA = 0.05 # opacity of drawn points
STEPS = 6000

ONE = 1./SIZE
STP = ONE/15.

RAD = 0.26 # radius of starting circle
FARL  = 0.13 # ignore "enemies" beyond this radius
NEARL = 0.02 # do not attempt to approach friends close than this

FRIENDSHIP_RATIO = 0.1 # probability of friendship dens
FRIENDSHIP_INITIATE_PROB = 0.05 # probability of friendship initation attempt

class Render(object):

  def __init__(self):

    self.__init_cairo()

  def __init_cairo(self):

    sur = cairo.ImageSurface(cairo.FORMAT_ARGB32,SIZE,SIZE)
    ctx = cairo.Context(sur)
    ctx.scale(SIZE,SIZE)
    ctx.set_source_rgb(*BACK)
    ctx.rectangle(0,0,1,1)
    ctx.fill()

    ctx.set_source_rgba(*FRONT)

    self.sur = sur
    self.ctx = ctx

  def dot(self,x,y):

    self.ctx.rectangle(x,y,ONE,ONE)
    self.ctx.fill()

 
class XRender(threading.Thread):

  def __init__(self, r, channel, path):
    threading.Thread.__init__(self)
    self.itt = 0

    self.filename = path+'/test_{:08d}_q{:s}.png'
    self.channel = channel

    self.render = Render()
    
    self.redis = r
    self.pubsub = self.redis.pubsub()
    self.pubsub.subscribe(channel)
  
  def work(self, item):

    d = loads(item['data'])
    print d['itt']
    for x,y in zip(d['x'],d['y']):
      self.render.dot(x,y)
  
  def run(self):

    for item in self.pubsub.listen():
      if item['data'] == 'KILL':
        self.pubsub.unsubscribe()
        print self, 'done!'
        break

      if item['data'] == 'DRAW':
        fn = self.filename.format(self.itt,self.channel)
        print self,'writing:',fn
        self.render.sur.write_to_png(fn)
        self.itt += 1
        continue

      if item['type'] == 'message':
        self.work(item)

def connections(r,itt,X,Y,F,A,R):

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
  
  r.publish('1', dumps({'x':xx,'y':yy,'itt':itt}))

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
        

if __name__ == "__main__":

  r = redis.Redis(host=HOST,port=PORT)

  xrender = XRender(r,'1','./img/')
  xrender.start()

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

      connections(r,itt,X,Y,F,A,R)

      if not itt % DRAW_ITT:
        print '**************************'

        r.publish('1','DRAW')


    #try:
      #for i in xrange(100000):
        #x = random()
        #y = random()
        #r.publish('1', dumps({'x':x,'y':y,'itt':i}))
  except Exception, e:
    print e
  finally:
    r.publish('1', 'KILL')

