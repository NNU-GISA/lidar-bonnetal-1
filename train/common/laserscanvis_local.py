#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import numpy as np
import cv2
from matplotlib import pyplot as plt
from common.laserscan import LaserScan, SemLaserScan


class LaserScanVis:
  """Class that creates and handles a visualizer for a pointcloud"""

  def __init__(self, scan, scan_names, label_names, offset=0,
               semantics=True, instances=False):
    print("scan {0}\nlabel {1}".format(scan_names, label_names))
    self.scan = scan
    self.scan_names = scan_names
    self.label_names = label_names
    self.offset = offset
    self.semantics = semantics
    self.instances = instances
    # sanity check
    if not self.semantics and self.instances:
      print("Instances are only allowed in when semantics=True")
      raise ValueError

    self.reset()
    self.update_scan()

  def reset(self):
    """ Reset. """
    # img canvas size
    self.multiplier = 1
    self.canvas_W = 1024
    self.canvas_H = 64
    if self.semantics:
      self.multiplier += 1
    if self.instances:
      self.multiplier += 1


  def get_mpl_colormap(self, cmap_name):
    cmap = plt.get_cmap(cmap_name)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

    return color_range.reshape(256, 3).astype(np.float32) / 255.0

  def update_scan(self):
    # first open data
    self.scan.open_scan(self.scan_names[self.offset])
    if self.semantics:
      self.scan.open_label(self.label_names[self.offset])
      self.scan.colorize()

    # then change names
    title = "scan " + str(self.offset) + " of " + str(len(self.scan_names))

    # then do all the point cloud stuff
    # plot scan
    power = 16
    # print()
    range_data = np.copy(self.scan.unproj_range)
    # print(range_data.max(), range_data.min())
    range_data = range_data**(1 / power)
    # print(range_data.max(), range_data.min())
    viridis_range = ((range_data - range_data.min()) /
                     (range_data.max() - range_data.min()) *
                     255).astype(np.uint8)
    viridis_map = self.get_mpl_colormap("viridis")
    viridis_colors = viridis_map[viridis_range]
    
    # now do all the range image stuff
    # plot range image
    data = np.copy(self.scan.proj_range)
    # print(data[data > 0].max(), data[data > 0].min())
    data[data > 0] = data[data > 0]**(1 / power)
    data[data < 0] = data[data > 0].min()
    # print(data.max(), data.min())
    data = (data - data[data > 0].min()) / \
        (data.max() - data[data > 0].min())
    # print(data.max(), data.min())
    # self.img_vis.set_data(data)
    # self.img_vis.update()

    if self.semantics:
      cv2.imwrite("/home/gaobiao/label.png", self.scan.proj_sem_color * 255)
      cv2.imwrite("/home/gaobiao/range.png", self.scan.proj_range)
      print("write to /home/gaobiao/ with shape {}".format(self.scan.proj_sem_color.shape))
      print(self.scan.proj_sem_color.max(), self.scan.proj_sem_color.min())
    #   self.sem_img_vis.set_data(self.scan.proj_sem_color[..., ::-1])
    #   self.sem_img_vis.update()

    # if self.instances:
    #   self.inst_img_vis.set_data(self.scan.proj_inst_color[..., ::-1])
    #   self.inst_img_vis.update()

#   # interface
#   def key_press(self, event):
#     self.canvas.events.key_press.block()
#     self.img_canvas.events.key_press.block()
#     if event.key == 'N':
#       self.offset += 1
#       self.update_scan()
#     elif event.key == 'B':
#       self.offset -= 1
#       self.update_scan()
#     elif event.key == 'Q' or event.key == 'Escape':
#       self.destroy()

#   def draw(self, event):
#     if self.canvas.events.key_press.blocked():
#       self.canvas.events.key_press.unblock()
#     if self.img_canvas.events.key_press.blocked():
#       self.img_canvas.events.key_press.unblock()

#   def destroy(self):
#     # destroy the visualization
#     self.canvas.close()
#     self.img_canvas.close()
#     vispy.app.quit()

#   def run(self):
    # vispy.app.run()
