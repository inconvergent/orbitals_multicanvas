#!/usr/bin/python
# -*- coding: utf-8 -*-

import redis
import threading

import cairo, Image
import numpy as np
from json import loads,dumps
from numpy.random import random

PORT = 6379
HOST = 'localhost'

SIZE = 1000
PIX = 1./SIZE

BACK = [1.]*3
FRONT = [0.5,0.1,0.1,1]

DRAW_ITT = 10000

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

    self.ctx.rectangle(x,y,PIX,PIX)
    self.ctx.fill()

 
class Listener(threading.Thread):

  def __init__(self, r, channels, render):
    threading.Thread.__init__(self)
    
    self.itt = 0
    self.file_name = './test_{:08d}.png'

    self.render = render
    self.redis = r
    self.pubsub = self.redis.pubsub()
    self.pubsub.subscribe(channels)
  
  def work(self, item):
  
    d = loads(item['data'])
    self.render.dot(d['x'],d['y'])

    if not d['i'] % DRAW_ITT:
      fname = self.file_name.format(d['i'])
      print fname
      self.render.sur.write_to_png(fname)
  
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

  client = Listener(r,['test'],Render())
  client.start()

  for i in xrange(100000):
    x = random()
    y = random()
    r.publish('test', dumps({'x':x,'y':y,'i':i}))
 
  r.publish('test', 'KILL')

