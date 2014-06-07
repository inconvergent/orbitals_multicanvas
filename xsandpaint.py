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


#COLOR_PATH = 'color/dark_cyan_white_black.gif'

PORT = 6379
HOST = 'localhost'

SIZE = 1000
SIZE = 20000 # size of png image

DRAW_ITT = 10000

PI = pi
TWOPI = pi*2.

NUM = 500 # number of nodes
MAXFS = 10 # max friendships pr node

BACK = 1. # background color 
GRAINS = 40
ALPHA = 0.05 # opacity of drawn points
STEPS = 10**7

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

    self.filename = path+'/test_{:08d}_q{:s}.png'
    self.channel = channel

    self.render = Render()
    
    self.redis = r
    self.pubsub = self.redis.pubsub()
    self.pubsub.subscribe(channel)
  
  def work(self, item):
  
    d = loads(item['data'])
    self.render.dot(d['x'],d['y'])

    if not d['i'] % DRAW_ITT:
      fn = self.filename.format(d['i'],self.channel)
      print self,'writing:',fn
      self.render.sur.write_to_png(fn)
  
  def run(self):

    for item in self.pubsub.listen():
      if item['data'] == 'KILL':
        self.pubsub.unsubscribe()
        print self, 'done!'
        break

      if item['type'] == 'message':
        self.work(item)
        

if __name__ == "__main__":

  r = redis.Redis(host=HOST,port=PORT)

  xrender = XRender(r,'1','./img/')
  xrender.start()

  try:
    for i in xrange(100000):
      x = random()
      y = random()
      r.publish('1', dumps({'x':x,'y':y,'i':i}))
  except:
    pass
  finally:
    r.publish('1', 'KILL')

