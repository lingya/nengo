"""
Common templates for constructing nodes that can communicate through ROS 
(Robot Operating System)
"""

from nengo.objects import Node
import rospy

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Wrench
from std_msgs.msg import String
import json

class RosPubNode( Node ):
  
  def __init__( self, name, topic, dimensions, msg_type, trans_fnc, period=30 ):
    """
    Parameters
    ----------
    name : str
        An arbitrary name for the object
    topic : str
        The name of the ROS topic that is being published to
    dimensions : int
        The number of input dimensions that this node will accept
    msg_type : msg
        The type of ROS message that will be published
    trans_fnc : callable
        A function that will transform the input into a valid ROS message of
        msg_type
    period : int
        How many time-steps to wait before publishing a message. A value of 1
        will publish at every time step
    """
    self.publishing_period = period
    self.counter = 0
    self.trans_fnc = trans_fnc
    self.msg_type = msg_type
    self.dimensions = dimensions
    self.topic = topic

    self.pub = rospy.Publisher( topic, msg_type )

    super( RosPubNode, self ).__init__( label=name, output=self.tick,
                                        size_in=dimensions, size_out=0 )

  def tick( self, t, values ):
    self.counter += 1
    if self.counter >= self.publishing_period:
      self.counter = 0
      msg = trans_fnc( values )
      self.pub.publish( msg )

class RosSubNode( Node ):
  
  def __init__( self, name, topic, dimensions, msg_type, trans_fnc ):
    """
    Parameters
    ----------
    name : str
        An arbitrary name for the object
    topic : str
        The name of the ROS topic that is being subscribed to
    dimensions : int
        The number of dimensions that this node will output
    msg_type : msg
        The type of ROS message that is being subscribed to
    trans_fnc : callable
        A function that will transform the ROS message of msg_type to an array
        with the appropriate number of dimensions to be used as output
    """
    self.trans_fnc = trans_fnc
    self.msg_type = msg_type
    self.dimensions = dimensions
    self.topic = topic

    self.rval = [0] * dimensions
    
    self.sub = rospy.Subscriber( topic, msg_type, self.callback )

    super( RosSubNode, self ).__init__( label=name, output=self.tick,
                                        size_in=0, size_out=dimensions )

  def callback( self, data ):
    self.rval = self.trans_fnc( data )

  def tick( self, t ):
    return self.rval


class SemanticCamera( RosSubNode ):
  """
  This node is active when specific targets are seen by a semantic camera in
  MORSE. Each target is represented by one dimension of the output, and a 0.0
  means the target is not seen by the camera, and a 1.0 means that the target is
  with the field of view of the camera.
  """
  
  #TODO: add optional weights and transform function for output
  def __init__( self, name, topic, targets ):
    """
    Parameters
    ----------
    name : str
        An arbitrary name for the object
    topic : str
        The name of the ROS topic that is being subscribed to
    targets : list
        List of str representing the names of the targets the camera is
        sensitive to
    """

    self.targets = targets
    self.dimensions = len( self.targets )

    def fn( data ):
      rval = [0] * self.dimensions
      string = data.data
      #TODO: put in error handling for malformed string
      str_val = json.loads( string )
      if len( str_val ) > 0:
        for i in str_val:
          if i['name'] in self.targets:
            rval[self.targets.index(i['name'])] = 1.0
            break
      
      return rval

    self.fn = fn

    super( SemanticCamera, self ).__init__( name=name, topic=topic,
                                        dimensions=self.dimensions,
                                        msg_type=String,trans_fnc=self.fn )
