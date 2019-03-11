import numpy as np

w = 8
h = 8


def loc_2D(loc_1D):
    # turn a 1D coordinate into a 2D coordinate
    return [loc_1D // w, loc_1D % w]


def loc_1D(loc_2D):
    # turn a 2D coordinate into a 1D coordinate
    return loc_2D[0] * w + loc_2D[1]


x = np.zeros([2, w * h * 9], dtype=np.int32)
y = np.zeros([2, w * h * 9], dtype=np.int32)

data = np.load('/nethome/yuefan/fanyue/dconv/layout.npy')
num_image = np.random.randint(0, 99)
print('plotting image number ', num_image)
c = 0
for key in range(len(data)):
    idx_list = data[key][num_image]
    key_loc = loc_2D(key)
    for i, loc in enumerate(idx_list):
        x[:, c*9 + i] = key_loc
        y[:, c*9 + i] = loc_2D(loc)
    c += 1

from bokeh.io import output_file, show
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure

output_file("brushing.html")

x1 = list(x[0, :])
y1 = list(x[1, :])
z1 = list(y[0, :])
z2 = list(y[1, :])

# create a column data source for the plots to share
source = ColumnDataSource(data=dict(x=x1, y=y1, z1=z1, z2=z2))

TOOLS = "box_select,lasso_select,help"

# create a new plot and add a renderer
left = figure(tools=TOOLS, plot_width=300, plot_height=300, title=None)
left.circle('x', 'y', source=source, color='black', radius=0.1)

# create another new plot and add a renderer
right = figure(tools=TOOLS, plot_width=300, plot_height=300, title=None)
right.circle('z1', 'z2', source=source, color='black', radius=0.1)

p = gridplot([[left, right]])

show(p)
