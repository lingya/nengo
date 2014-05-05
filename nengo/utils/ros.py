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
      msg = self.trans_fnc( values )
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


class SemanticCameraNode( RosSubNode ):
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

    super( SemanticCameraNode, self ).__init__( name=name, topic=topic,
                                                dimensions=self.dimensions,
                                                msg_type=String, trans_fnc=self.fn )

class ForceTorqueNode( RosPubNode ):
  """
  This node publishes a force and torque
  """
  
  #TODO: add optional weights and transform function for input
  def __init__( self, name, topic, 
                attributes=[True, True, True, True, True, True] ):
    """
    Parameters
    ----------
    name : str
        An arbitrary name for the object
    topic : str
        The name of the ROS topic that is being subscribed to
    attributes : list
        List of boolean representing the which dimensions of the Wrench message
        are being published. All others will be left as zero.
        For Example, [True, False, False, False, False, True] will publish the
        the force in the x direction, as well as the torque in the z direction.
        Since only two fields of the message are being used, the node will have
        only two dimensions. The format is:
          [force.x, force.y, force.z, torque.x, torque.y, torque.z]
    """

    #TODO: change this from semantic stuff to forcetorque stuff
    self.attributes = attributes
    self.dimensions = attributes.count( True )

    def fn( values ):
      wrench = Wrench()
      index = 0
      if self.attributes[0]:
        wrench.force.x = values[index]
        index += 1
      if self.attributes[1]:
        wrench.force.y = values[index]
        index += 1
      if self.attributes[2]:
        wrench.force.z = values[index]
        index += 1
      if self.attributes[3]:
        wrench.torque.x = values[index]
        index += 1
      if self.attributes[4]:
        wrench.torque.y = values[index]
        index += 1
      if self.attributes[5]:
        wrench.torque.z = values[index]
        index += 1
      return wrench

    self.fn = fn

    super( ForceTorqueNode, self ).__init__( name=name, topic=topic,
                                             dimensions=self.dimensions,
                                             msg_type=Wrench, trans_fnc=self.fn )

class OdometryNode( RosSubNode ):
  """
  This node reads odometry data
  """
  
  #TODO: add optional weights and transform function for output
  #TODO: need to come up with a better way of specifying 'attributes' as well as
  #      a better name for this parameter
  def __init__( self, name, topic, attributes ):
    """
    Parameters
    ----------
    name : str
        An arbitrary name for the object
    topic : str
        The name of the ROS topic that is being subscribed to
    attributes : list
        List of boolean representing the which dimensions of the Odometry message
        are being published. All others will be left as zero.
    """

    self.attributes = attributes
    self.dimensions = attributes.count( True )

    def fn( data ):
      rval = [0] * self.dimensions

      index = 0

      if self.attributes[ 0 ]:
        rval[index] = data.pose.pose.position.x
        index += 1
      if self.attributes[ 1 ]:
        rval[index] = data.pose.pose.position.y
        index += 1
      if self.attributes[ 2 ]:
        rval[index] = data.pose.pose.position.z
        index += 1

      if self.attributes[ 3 ]:
        rval[index] = data.twist.twist.linear.x
        index += 1
      if self.attributes[ 4 ]:
        rval[index] = data.twist.twist.linear.y
        index += 1
      if self.attributes[ 5 ]:
        rval[index] = data.twist.twist.linear.z
        index += 1

      # FIXME: ROS outputs orientation in quaternions, need to account for this
      if self.attributes[ 6 ]:
        rval[index] = data.pose.pose.orientation.x
        index += 1
      if self.attributes[ 7 ]:
        rval[index] = data.pose.pose.orientation.y
        index += 1
      if self.attributes[ 8 ]:
        rval[index] = data.pose.pose.orientation.z
        index += 1

      if self.attributes[ 9 ]:
        rval[index] = data.twist.twist.angular.x
        index += 1
      if self.attributes[ 10 ]:
        rval[index] = data.twist.twist.angular.y
        index += 1
      if self.attributes[ 11 ]:
        rval[index] = data.twist.twist.angular.z
        index += 1
      
      return rval

    self.fn = fn

    super( OdometryNode, self ).__init__( name=name, topic=topic,
                                          dimensions=self.dimensions,
                                          msg_type=Odometry, trans_fnc=self.fn )
