#!/usr/bin/env python

"""
__author__ = "Lech Szymanski"
__copyright__ = "Copyright 2019, COSC343"
__license__ = "GPL"
__version__ = "2.0.1"
__maintainer__ = "Lech Szymanski"
__email__ = "lechszym@cs.otago.ac.nz"
"""

from ctypes import *
import numpy as np
import time
import sys

# Load the Engine lib
if sys.platform == "linux" or sys.platform == "linux2":
   lib = cdll.LoadLibrary('./libcosc343worldcc.so')
elif sys.platform == "darwin":
   lib = cdll.LoadLibrary('./libcosc343worldcc.dylib')
elif sys.platform == "win32":
   lib = CDLL('./cosc343worldcc.dll')
   
   
# Declare function signatures from the Engine lib for the Creatures class
lib.Creature_new.restype  = c_void_p
lib.Creature_new.argtypes = [c_void_p, py_object]
lib.Creature_delete.restype  = None
lib.Creature_delete.argtypes = [c_void_p]
lib.Creature_numPercepts.restype = c_int32
lib.Creature_numPercepts.argtypes = [c_void_p]
lib.Creature_numActions.restype = c_int32
lib.Creature_numActions.argtypes = [c_void_p]
lib.Creature_getPercept.restype = c_int32
lib.Creature_getPercept.argtypes = [c_void_p, c_int32]
lib.Creature_setAction.restype = None
lib.Creature_setAction.argtypes = [c_void_p, c_int32, c_float]
lib.Creature_getEnergy.restype = c_int32
lib.Creature_getEnergy.argtypes = [c_void_p]
lib.Creature_isDead.restype = c_bool
lib.Creature_isDead.argtypes = [c_void_p]
lib.Creature_timeOfDeath.restype = c_int32
lib.Creature_timeOfDeath.argtypes = [c_void_p]
lib.Creature_getx.restype = c_int32
lib.Creature_getx.argtypes = [c_void_p]
lib.Creature_gety.restype = c_int32
lib.Creature_gety.argtypes = [c_void_p]

# Declare function signatures from the Engine lib for the World class
lib.World_new.restype  = c_void_p
lib.World_new.argtypes = [c_int32, c_int32, c_bool]
lib.World_delete.restype  = None
lib.World_delete.argtypes = [c_void_p]
lib.World_gridSize.restype  = c_int32
lib.World_gridSize.argtypes = [c_void_p]
lib.World_maxNumCreatures.restype  = c_int32
lib.World_maxNumCreatures.argtypes = [c_void_p]
lib.World_numCreaturePercepts.restype = c_int32
lib.World_numCreaturePercepts.argtypes = [c_void_p]
lib.World_numCreatureActions.restype = c_int32
lib.World_numCreatureActions.argtypes = [c_void_p]
lib.World_resetCreatures.restype = None
lib.World_resetCreatures.argtypes = [c_void_p]
lib.World_addCreature.restype = None
lib.World_addCreature.argtypes = [c_void_p, c_void_p]
lib.World_evaluate.restype = None
lib.World_evaluate.argtypes = [c_void_p, c_int32]
lib.World_vis_num.restype = c_int32
lib.World_vis_num.argtypes = [c_void_p, c_int32]
lib.World_vis_numTurns.restype = c_int32
lib.World_vis_numTurns.argtypes = [c_void_p]
lib.World_vis.restype = c_int32
lib.World_vis.argtypes = [c_void_p, c_int32, c_int32, c_int32, c_int32]

# Agent callback function (invoked from the Engine when a crature
# needs to take action)
@CFUNCTYPE(None, py_object)
def agent_callback(creature):
    creature.internal_AgentFunction()

# This is a creature class that your EvolvingCreature needs to inherit from
# This class wraps the _cCreature class which was implemented in C.
class Creature(object):

    # Constructor
    def __init__(self):
        self.obj = lib.Creature_new(agent_callback, py_object(self))

    def __del__(self):
        lib.Creature_delete(self.obj)

    # Returns the energy of the creature
    def getEnergy(self):
        return lib.Creature_getEnergy(self.obj)

    # Returns boolean indicating whether creature is alive or dead
    def isDead(self):
        return lib.Creature_isDead(self.obj)

    # Returns time of death of the creature (in turns)
    def timeOfDeath(self):
        return lib.Creature_timeOfDeath(self.obj)

    # Your child class must override this method, where the
    # mapping of percepts to actions is implemented
    def AgentFunction(self, percepts, nActions):
        print("Your EvolvingCreature needs to override the AgentFunction method!")
        sys.exit(-1)

    # Agent function evoked from the simulation of the world implemented in C.
    # This method translates the percepts to python list, and translates back
    # the list representing the actions into C format.
    def internal_AgentFunction(self):

        # Get the number of percepts and actions
        nPercepts = lib.Creature_numPercepts(self.obj)
        nActions = lib.Creature_numActions(self.obj)

        # Create lists of percepts
        percepts = np.zeros((nPercepts))
        for i in range(nPercepts):
            percepts[i] = lib.Creature_getPercept(self.obj, i)

        # Execute the AgentFunction method that needs to be implemented
        # by the EvolvingCreature.  Pass in the list of percepts and
        # specify the number of actions expected.
        actions = self.AgentFunction(percepts, nActions)

        if not isinstance(actions, list) or len(actions) != nActions:
            print("Error!  Expecting the actions returned from the AgentFunction to be a list of %d numbers." % nActions)

        # Translate actions and feed it back to the engine
        for i in range(nActions):
            lib.Creature_setAction(self.obj, i, float(actions[i]))

# Wrapper class for _cWorld which implements the engine for the simulation
class World(object):

   # Initialisation wrapper with some defaults for world type, grid size
   # and repeatability setting.
   def __init__(self, worldType=1, gridSize=24, repeatable=False):
      self.obj = lib.World_new(worldType, gridSize, repeatable)
      self.ph = None
      self.worldType = worldType

   def __del__(self):
       lib.World_delete(self.obj)


   # Returns the number of creaturs in the population
   def maxNumCreatures(self):
      return lib.World_maxNumCreatures(self.obj)

   # Returns the number of percepts per creatures
   def numCreaturePercepts(self):
      return lib.World_numCreaturePercepts(self.obj)

   # Returns the number of actions
   def numCreatureActions(self):
      return lib.World_numCreatureActions(self.obj)

   # Feed the next generation of creatures to the simulation
   #
   # Input: population - a list of creatures for simulation
   def setNextGeneration(self, population):
      lib.World_resetCreatures(self.obj)
      for i in range(len(population)):
         lib.World_addCreature(self.obj, population[i].obj)
         #self.addCreature(population[i])

   def evaluate(self, numTurns):
      lib.World_evaluate(self.obj, numTurns)

   # Animation of the simulation
   #
   # Input: titleStr - title string of the simulation
   #        speed - of the simulation: can be 'slow', 'normal' or 'fast'
   def show_simulation(self, titleStr = "", speed='normal'):
      import pygame
      gridSize = lib.World_gridSize(self.obj)
      left_frame = 100

      pygame.init()

      size = width, height = 720, 480
      WHITE = (255, 255, 255)
      BLACK = 0, 0, 0

      if speed == "normal":
          frameTurns = 20
          nSteps = 10
      elif speed == "fast":
          frameTurns = 1
          nSteps = 5
      elif speed == "slow":
          frameTurns = 40
          nSteps = 10

      screen = pygame.display.set_mode(size)

      unit = int(np.min([width-left_frame, height])/gridSize)

      im_strawbs = [pygame.image.load('images/strawberry-green.png'),
                    pygame.image.load('images/strawberry-red.png')
                   ]

      im_creatures = [pygame.image.load('images/smiley_happy.png'),
                      pygame.image.load('images/smiley_hungry.png'),
                      pygame.image.load('images/smiley_sick.png')
                     ]

      for i in range(len(im_strawbs)):
          im_strawbs[i] = pygame.transform.scale(im_strawbs[i], (unit, unit))

      for i in range(len(im_creatures)):
          im_creatures[i] = pygame.transform.scale(im_creatures[i], (unit, unit))

      im_monster = pygame.transform.scale(pygame.image.load("images/monster.png"), (unit, unit))

      nTurns = lib.World_vis_numTurns(self.obj)
      stepDiff = 1.0/float(nSteps)

      nFood = lib.World_vis_num(self.obj, 0)
      nCreatures = lib.World_vis_num(self.obj, 1)
      nMonsters = lib.World_vis_num(self.obj, 2)

      nBodies = [nFood, nCreatures, nMonsters]

      halfSteps = int(np.floor(nSteps/2))

      for t in range(1, nTurns + 1):

          pygame.display.set_caption("World %d, %s (turn %d)" % (self.worldType, titleStr, t))

          for k in range(nSteps):

              for event in pygame.event.get():
                  if event.type == pygame.QUIT: sys.exit()


              screen.fill(WHITE)

              for i in range(gridSize + 1):
                 pygame.draw.line(screen, BLACK, [left_frame, i*unit], [left_frame+(gridSize*unit), i*unit])
                 pygame.draw.line(screen, BLACK, [left_frame+(i*unit), 0], [left_frame+(i*unit), gridSize * unit])

              for type in range(3):
                  for i in range(nBodies[type]):
                      x = lib.World_vis(self.obj, type, 0, i, t)
                      y = lib.World_vis(self.obj, type, 1, i, t)
                      s = lib.World_vis(self.obj, type, 2, i, t)

                      xprev = lib.World_vis(self.obj, type, 0, i, t-1)
                      yprev = lib.World_vis(self.obj, type, 1, i, t-1)

                      xshift = xprev-x
                      if np.abs(xshift)<=1:
                          xdiff = (x - xprev) * k * stepDiff
                      elif k <= halfSteps:
                          xdiff = np.sign(xshift) * k * stepDiff
                      else:
                          xdiff = -np.sign(xshift) * k * stepDiff
                          xprev = x

                      yshift = yprev - y
                      if np.abs(yshift) <= 1:
                          ydiff = (y - yprev) * k * stepDiff
                      elif k <= halfSteps:
                          ydiff = np.sign(yshift) * k * stepDiff
                      else:
                          ydiff = -np.sign(yshift) * k * stepDiff
                          yprev = y

                      if type==0:
                          if s >= 0 and s <= 1:
                              obj_loc = pygame.Rect(left_frame + (x * unit), y * unit, unit, unit)
                              obj_im = im_strawbs[s]
                              screen.blit(obj_im, obj_loc)

                      elif type==1:
                          if s > 0:
                              obj_im = im_creatures[s-1]
                              obj_loc = pygame.Rect(left_frame + (xprev + xdiff) * unit, (yprev + ydiff) * unit, unit,
                                                    unit)
                              screen.blit(obj_im, obj_loc)


                      elif type==2:
                          obj_loc = pygame.Rect(left_frame+(xprev + xdiff) * unit, (yprev + ydiff) * unit, unit, unit)
                          screen.blit(im_monster, obj_loc)

              pygame.display.flip()
              pygame.time.delay(frameTurns)
      pygame.display.quit()
      pygame.quit()
