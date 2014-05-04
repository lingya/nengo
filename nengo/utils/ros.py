"""
Common templates for constructing nodes that can communicate through ROS 
(Robot Operating System)
"""

from nengo.objects import Node
import rospy

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

    self.output = [0] * dimensions
    
    self.sub = rospy.Subscriber( topic, msg_type, self.callback )

    super( RosSubNode, self ).__init__( label=name, output=self.tick,
                                        size_in=0, size_out=dimensions )

  def callback( self, data ):
    self.output = trans_fnc( data )


  def tick( self, t, values ):
    return self.output
