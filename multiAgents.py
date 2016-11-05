# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """


  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    "Add more of your code here if you want to"

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    #print successorGameState

    newFood = successorGameState.getFood().asList()
    output = 0

    fMin = 99999
    for food in newFood:
        u = util.manhattanDistance(newPos, food)
        if u < fMin and u != 0:
            fMin = u

    gMin = 99999
    for ghostState in newGhostStates:
        g = util.manhattanDistance(newPos, ghostState.getPosition())
        if g < gMin:
            gMin = g

    if gMin == 0 or gMin > 20:
        gMin = -1000

    return gMin/fMin + successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

#Based on algorithm from slides
class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    return self.miniMaxStarter(gameState)

  #Pac-man

  #ROOT Node separated out because value needs to be converted to action
  def miniMaxStarter(self, gameState):
      bestAction = ""
      v = -100000
      for action in gameState.getLegalActions():
          prev = v
          v = max(v, self.minValue(gameState.generateSuccessor(0, action), 0, 1))
          if v > prev:
              bestAction = action

      return bestAction

  #Max choice
  def maxValue(self, gameState, depth):

      v = -100000#sys.minint

      if gameState.isWin() or gameState.isLose() or depth == self.depth - 1: #bottom has been reached, evaluate
          return self.evaluationFunction(gameState)

      #Returns value for each action
      for action in gameState.getLegalActions(0):
          if action != Directions.STOP:
              v = max(v, self.minValue(gameState.generateSuccessor(0, action), depth, 1))

      return v

  #Min Choice
  def minValue(self, gameState, depth, numGhost):

      if gameState.isWin() or gameState.isLose() or depth == self.depth - 1: #bottom has been reached, evaluate
          return self.evaluationFunction(gameState)

      v = 100000#sys.maxint

      #Run through all ghosts and get their mins, once last ghost is reached and depth hasn't been achieved, max is called
      for action in gameState.getLegalActions(numGhost):
          if action != Directions.STOP:
              if numGhost == gameState.getNumAgents() - 1:
                  v = min(v, self.maxValue(gameState.generateSuccessor(numGhost, action), depth + 1))
              else:
                  v = min(v, self.minValue(gameState.generateSuccessor(numGhost, action), depth, numGhost + 1))
      return v

#Based on algorithm from slides
class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    return self.alphaBetaStarter(gameState)

  #Starter Function for root node to convert score into action
  def alphaBetaStarter(self, gameState):
    bestAction = ""
    v = alpha = -100000
    beta = 10000
    for action in gameState.getLegalActions():
        prev = v
        v = max(v, self.minValue(gameState.generateSuccessor(0, action), alpha, beta, 0, 1)) #calls root children
        if v > prev:
            bestAction = action
        if v >= beta:
            return bestAction
        alpha = max(alpha, v)

    return bestAction

  def maxValue(self, gameState, alpha, beta, depth):

      v = -100000

      if gameState.isWin() or gameState.isLose() or depth == self.depth - 1: #bottom has been reached, evaluate
          return self.evaluationFunction(gameState)

      for action in gameState.getLegalActions():
          if action != Directions.STOP:
              v = max(v, self.minValue(gameState.generateSuccessor(0, action), alpha, beta, depth, 1))

              if v >= beta:
                  return v

              alpha = max(alpha, v)

      return v

  def minValue(self, gameState, alpha, beta, depth, numGhost):
      if gameState.isWin() or gameState.isLose() or depth == self.depth - 1:
          return self.evaluationFunction(gameState)

      v = 100000

      for action in gameState.getLegalActions(numGhost):
          if action != Directions.STOP:
              if numGhost != gameState.getNumAgents()-1:
                  v = min(v, self.maxValue(gameState.generateSuccessor(numGhost, action), alpha, beta, depth + 1))

              else:
                  v = min(v, self.minValue(gameState.generateSuccessor(numGhost, action), alpha, beta, depth, numGhost + 1))

              if v <= alpha:
                  return v
              beta = min(beta, v)
      return v

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()

    return self.maxValue(gameState, 0)

  def maxValue(self, gameState, depth):

    x = 10
    v = -100000
    bestAction = ""

    for action in gameState.getLegalActions(0):
          if action != Directions.STOP:
              x = self.expValue(gameState.generateSuccessor(0, action), depth, 1)

              if x > v:
                  v = x
                  bestAction = action

    if depth == 0:
          return bestAction
    else:
          return v

  def expValue(self, gameState, depth, numGhosts):

      if depth == self.depth - 1:
          return self.evaluationFunction(gameState)

      v = 100000

      #l = len(gameState.getLegalActions(gameState)
      print gameState.getLegalActions(gameState)

      for action in gameState.getLegalActions():
          if action != Directions.STOP:
            if numGhost != gameState.getNumAgents()-1:
              v = min(v, self.maxValue(gameState.generateSuccessor(numGhost, action), depth + 1))
              print "BLA"

          else:
            v = min(v, self.expValue(gameState.generateSuccessor(numGhost, action), depth, numGhost + 1))
            print "BLA"

      return v

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  "*** YOUR CODE HERE ***"
  #util.raiseNotDefined()

  gMin = 99999
  fMin = 99999
  for action in currentGameState.getLegalActions():
      successorGameState = currentGameState.generatePacmanSuccessor(action)
      newPos = successorGameState.getPacmanPosition()
      oldFood = currentGameState.getFood()
      newFood = successorGameState.getFood()
      newGhostStates = successorGameState.getGhostStates()
      oldGhostStates = currentGameState.getGhostStates()
      newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

      for ghostState in newGhostStates:
            g = util.manhattanDistance(newPos, ghostState.getPosition())
            if g < gMin:
                gMin = g

      newFoodCount = 0
      for food in newFood:
            newFoodCount += 1
            u = util.manhattanDistance(newPos, food)
            if u < fMin and u != 0:
                fMin = u

      oldFoodCount = 0
      for food in oldFood:
            oldFoodCount += 1

      print "GHOST_SCORE: ", gMin
      print "ACTION: ", action
      print "newfood: ", newFoodCount

      if gMin > 3:
          fMin *= 100

      print fMin-gMin
  return fMin-gMin

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    actionB = Directions.STOP
    prev = 0
    for action in gameState.getLegalActions():
        x = evaluationFunction(self, gameState, action)

        if x > prev:
            prev = x
            actionB = action

    return actionB
